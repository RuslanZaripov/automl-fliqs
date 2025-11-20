import os
import numpy as np
import torch
import random
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)
import argparse
from datetime import datetime
from trainer.reinforce_trainer import ReinforceTrainer, ReinforceConfig
from eval.eval_parse import (
    create_arg_parser,
    SUPPORTED_GLUE_TASKS,
    ALL_MODEL_MAP,
    VISON_MODEL_MAP,
    LANGUAGE_MODEL_MAP,
)
from eval.eval_partitions import BERT_BLOCK_PARTITIONS
from quant.quant_module import model_cost, model_weight_percentiles
from functools import partial

from eval.transformer_utils import (
    get_transformer_model,
    quantize_transformers,
    get_transformer_num_labels,
    glue_tokenizer,
    glue_compute_metrics,
)
from eval.vision_utils import get_vision_model, quantize_vision_model

# This import is needed for globals()
from cost.cost import tunas_abs, soft_exponential, hard_exponential


def get_validation_key(task: str):
    """Get the key for the validation set for a given task"""
    # For GLUE benchmarks, the test set is withheld so we evaluate on the validation set
    # MNLI has two different validation sets so we use the matched validation set
    if task == "mnli":
        return "validation_matched"
    elif task == "cifar10" or task == "cifar100" or task == "imagenet-1k":
        return "test"
    else:
        return "validation"


def get_raw_datasets(args):
    if args.task in SUPPORTED_GLUE_TASKS:
        return load_dataset("glue", args.task)
    else:
        return load_dataset(args.task)


def preprocess_data(args, raw_datasets, model_type):
    processor, tokenizer, data_collator, compute_metrics = None, None, None, None

    if model_type in VISON_MODEL_MAP.values():
        processor = AutoImageProcessor.from_pretrained(model_type)

        def preprocess_image(data, processor):
            inputs = processor(data["img"], return_tensors="pt")
            inputs["labels"] = torch.tensor(data["label"])
            return inputs

        preprocess = partial(preprocess_image, processor=processor)

    elif model_type in LANGUAGE_MODEL_MAP.values():
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        if "gpt2" in model_type:
            tokenizer.pad_token = tokenizer.eos_token
        preprocess = partial(glue_tokenizer, task=args.task, tokenizer=tokenizer)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        compute_metrics = lambda eval_preds: glue_compute_metrics(
            eval_preds, task=args.task
        )

    else:
        raise ValueError(f"Unsupported model: {model_type}")

    processed_datasets = raw_datasets.map(preprocess, batched=True)

    return processed_datasets, data_collator, tokenizer, compute_metrics


def get_training_args(args, run_name, output_dir):
    training_args = TrainingArguments(
        output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        max_steps=args.max_steps,
        logging_dir=output_dir + "/runs",
        run_name=run_name,
        logging_steps=args.log_steps,
        save_strategy=args.save_strategy,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optimizer,
    )
    return training_args


def quantize_model(args, model, model_type):
    if args.weight_bits > 0 or args.activation_bits > 0:
        if model_type in VISON_MODEL_MAP.values():
            model, q_modules = quantize_vision_model(
                model,
                weight_bits=args.weight_bits,
                activation_bits=args.activation_bits,
                per_channel=args.per_channel,
                act_range_percentile=args.act_range_percentile,
                weight_range_percentile=args.weight_range_percentile,
            )
        elif model_type in LANGUAGE_MODEL_MAP.values():
            model, q_modules = quantize_transformers(
                args.model_type,
                model,
                per_channel=args.per_channel,
                per_token=args.per_token,
                weight_bits=args.weight_bits,
                activation_bits=args.activation_bits,
                quantize_self_attention=args.quantize_self_attention,
                quantize_self_output=args.quantize_self_output,
                quantize_intermediate=args.quantize_intermediate,
                quantize_output=args.quantize_output,
                quantize_pooler=args.quantize_pooler,
                quantize_classifier=args.quantize_classifier,
                act_range_percentile=args.act_range_percentile,
                weight_range_percentile=args.weight_range_percentile,
            )
        else:
            raise ValueError(f"Unsupported model: {model_type}")
    else:
        q_modules = {}
    return model, q_modules


def get_trainer(
    args,
    training_args,
    model,
    q_modules,
    processed_datasets,
    data_collator,
    tokenizer,
    compute_metrics,
    output_dir,
):
    validation_key = get_validation_key(args.task)
    train_key = "train"
    log_dir = output_dir + "/log"
    if args.search:
        if args.block_search:
            block_partitions = BERT_BLOCK_PARTITIONS
        else:
            block_partitions = None
        search_options = [int(x) for x in args.search_space.split(",")]
        cost_function = globals()[args.cost_function]
        reinforce_config = ReinforceConfig(
            delay_steps=args.delay_steps,
            cost_beta=args.cost_beta,
            cost_target=args.cost_target,
            learning_rate=args.rl_learning_rate,
            cost_function=cost_function,
            max_entropy_beta=args.max_entropy_beta,
            entropy_schedule_type=args.entropy_schedule_type,
            weight_only=args.weight_only_search,
            single_decision=args.single_decision,
            block_partitions=block_partitions,
            cost_type=args.cost_type,
        )
        trainer = ReinforceTrainer(
            q_modules,
            search_options,
            model,
            training_args,
            reinforce_config,
            log_dir=log_dir,
            train_dataset=processed_datasets[train_key],
            eval_dataset=processed_datasets[validation_key],
            data_collator=data_collator,
            tokenizer=tokenizer,
            seed=args.seed,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_datasets[train_key],
            eval_dataset=processed_datasets[validation_key],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    return trainer


def save_outputs(args, eval_metrics, output_dir, model, trainer):
    with open(os.path.join(output_dir, "output.txt"), "w") as f:
        # Write the command-line arguments
        f.write("Command-line Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")

        # Save the final evaluation metrics to eval.txt
        for key, value in eval_metrics.items():
            f.write(f"{key}: {value}\n")
        # Write the model memory cost
        cost = model_cost(model, cost_type=args.cost_type)
        f.write(f"Cost (QLinear Only): {cost:.5f}\n")
        # Write percentiles of weights
        percentiles = model_weight_percentiles(model)
        for key, value in percentiles.items():
            f.write("Weight Percentiles:")
            f.write(f"{key}: {value}\n")
        # Write self.best_bits and self.best_reward if they exist
        if args.search:
            f.write("\nLast Bits:\n")
            for key, value in trainer.last_bits.items():
                f.write(f"{key}: {value}\n")


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args: argparse.Namespace) -> None:
    # Set the seed for reproducibility
    set_seed(args.seed)
    raw_datasets = get_raw_datasets(args)

    # Only use a subset of the data for testing
    if args.use_subset:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(5))

    model_type = ALL_MODEL_MAP[args.model_type]
    processed_datasets, data_collator, tokenizer, compute_metrics = preprocess_data(
        args, raw_datasets, model_type
    )

    current_date_string = datetime.now().strftime("%Y-%m-%d")
    if args.exp_name:
        run_name = args.exp_name
        output_dir = args.output_root_dir + "/" + args.task + "/" + args.exp_name
    else:
        run_name = current_date_string
        output_dir = args.output_root_dir + "/" + args.task + "/" + current_date_string

    training_args = get_training_args(args, output_dir, run_name)
    if model_type in LANGUAGE_MODEL_MAP.values():
        num_labels = get_transformer_num_labels(args.task)
        model = get_transformer_model(model_type, num_labels)
    elif model_type in VISON_MODEL_MAP.values():
        model = get_vision_model(model_type)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Optionally convert to half precision
    if args.fp16:
        model = model.half()

    # Quantize the model
    model, q_modules = quantize_model(args, model, model_type)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model config
    with open(os.path.join(output_dir, "model.txt"), "w") as f:
        f.write(repr(model))

    # Save the args to config.txt
    with open(os.path.join(output_dir, "config.txt"), "w") as f:
        f.write(str(args))

    # Get the trainer
    trainer = get_trainer(
        args,
        training_args,
        model,
        q_modules,
        processed_datasets,
        data_collator,
        tokenizer,
        compute_metrics,
        output_dir,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()

    # Save the outputs
    save_outputs(args, eval_metrics, output_dir, model, trainer)
    print(eval_metrics)


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    main(args)
