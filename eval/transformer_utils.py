from typing import Dict, Any, Tuple
import numpy as np
from transformers import PreTrainedModel
from quant.quant_linear import QuantLinear
import evaluate
from eval.eval_parse import SUPPORTED_GLUE_TASKS

from transformers import AutoModelForSequenceClassification

from transformers.models.bert.modeling_bert import (
    BertSelfOutput,
    BertSelfAttention,
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertForSequenceClassification,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaSelfOutput,
    RobertaSelfAttention,
    RobertaIntermediate,
    RobertaOutput,
    RobertaPooler,
    RobertaForSequenceClassification,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP,
    GPT2ForSequenceClassification,
    GPT2Attention,
)


def get_transformer_model(model_type: str, num_labels: int) -> PreTrainedModel:
    if "gpt2" in model_type:
        model = GPT2ForSequenceClassification.from_pretrained(
            model_type, num_labels=num_labels
        )
        model.config.pad_token_id = model.config.eos_token_id
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_type, num_labels=num_labels
        )

    return model


def get_transfomer_submodules(model_type: str):
    """Get the transformer submodule class types for a given model type"""
    if "bert" in model_type and not model_type == "roberta":
        return (
            BertSelfAttention,
            BertSelfOutput,
            BertIntermediate,
            BertOutput,
            BertPooler,
            BertForSequenceClassification,
        )
    elif model_type == "roberta":
        return (
            RobertaSelfAttention,
            RobertaSelfOutput,
            RobertaIntermediate,
            RobertaOutput,
            RobertaPooler,
            RobertaForSequenceClassification,
        )
    elif "gpt2" in model_type:
        return GPT2Attention, GPT2MLP
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def quantize_transformers(
    model_type: str,
    model: PreTrainedModel,
    per_channel: bool = False,
    per_token: bool = False,
    weight_bits: int = 8,
    activation_bits: int = 8,
    quantize_self_attention: bool = True,
    quantize_self_output: bool = True,
    quantize_intermediate: bool = True,
    quantize_output: bool = True,
    quantize_pooler: bool = False,
    quantize_classifier: bool = False,
    act_range_percentile: float = 0,
    weight_range_percentile: float = 0,
) -> Tuple[PreTrainedModel, Dict[str, QuantLinear]]:
    """Quantize all nn.Linear layers within a given transformer model"""
    quantized_modules = {}

    model_classes = get_transfomer_submodules(model_type)

    def replace_layer_with_quantized(layer, layer_name, transpose=False):
        new_layer = QuantLinear(
            layer,
            weight_bit=weight_bits,
            activation_bit=activation_bits,
            per_channel=per_channel,
            per_token=per_token,
            act_percentile=act_range_percentile,
            weight_percentile=weight_range_percentile,
            transpose=transpose,
        )
        quantized_modules[layer_name] = new_layer
        return new_layer

    if "gpt2" in model_type:
        for name, m in model.named_modules():
            if isinstance(m, GPT2Attention):
                for layer_name in ["c_attn", "c_proj"]:
                    layer = getattr(m, layer_name)
                    setattr(
                        m,
                        layer_name,
                        replace_layer_with_quantized(
                            layer, f"{name}.{layer_name}", transpose=True
                        ),
                    )
            elif isinstance(m, GPT2MLP):
                for layer_name in ["c_fc", "c_proj"]:
                    layer = getattr(m, layer_name)
                    setattr(
                        m,
                        layer_name,
                        replace_layer_with_quantized(
                            layer, f"{name}.{layer_name}", transpose=True
                        ),
                    )
    elif "bert" in model_type:
        for name, m in model.named_modules():
            if quantize_self_attention and isinstance(m, model_classes[0]):
                for qkv_name in ["query", "key", "value"]:
                    qkv_layer = getattr(m, qkv_name)
                    setattr(
                        m,
                        qkv_name,
                        replace_layer_with_quantized(qkv_layer, f"{name}.{qkv_name}"),
                    )

            if isinstance(m, tuple(model_classes[1:])):
                layer_name = name + ".dense"
                if (
                    quantize_self_output
                    and isinstance(m, model_classes[1])
                    or quantize_intermediate
                    and isinstance(m, model_classes[2])
                    or quantize_output
                    and isinstance(m, model_classes[3])
                    or quantize_pooler
                    and isinstance(m, model_classes[4])
                ):
                    m.dense = replace_layer_with_quantized(m.dense, layer_name)
                elif quantize_classifier and isinstance(m, model_classes[5]):
                    layer_name = name + ".classifier"
                    m.classifier = replace_layer_with_quantized(
                        m.classifier, layer_name
                    )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, quantized_modules


def glue_compute_metrics(
    eval_preds: Tuple[np.ndarray, np.ndarray], task: str
) -> Dict[str, Any]:
    """Compute metrics for GLUE tasks"""
    if task in SUPPORTED_GLUE_TASKS:
        metric = evaluate.load("glue", task)
    else:
        metric = evaluate.load(task)

    logits, labels = eval_preds
    if task in ["stsb"]:
        predictions = logits[:, 0]
    else:
        predictions = np.argmax(logits, axis=-1)

    if task in [
        "cola",
        "sst2",
        "mrpc",
        "qqp",
        "qnli",
        "rte",
        "wnli",
        "mnli",
        "boolq",
        "arc-e",
        "arc-c",
        "openbookqa",
    ]:
        return metric.compute(predictions=predictions, references=labels)
    # stsb handled separately because it is a regression task
    elif task in ["stsb"]:
        return metric.compute(
            predictions=predictions,
            references=labels,
            round_mode="half_to_even",
            round_ndigits=3,
        )
    # piqa handled separately because of the choice of answer index
    elif task in ["piqa"]:
        return metric.compute(predictions=predictions, references=labels.squeeze())
    else:
        raise ValueError(f"Unsupported task: {task}")


def get_transformer_num_labels(task: str):
    """Get the number of labels for a given task"""
    if task in ["cola", "sst2", "mrpc", "qqp", "qnli", "rte", "wnli", "boolq"]:
        num_labels = 2
    elif task == "mnli":
        num_labels = 3
    elif task == "piqa":
        num_labels = 2  # Assuming there are only two solutions per problem
    else:
        raise ValueError(f"Unsupported task: {task}")

    return num_labels


def glue_tokenizer(example: Dict[str, Any], task: str, tokenizer) -> Dict[str, Any]:
    if task in ["cola", "sst2", "stsb"]:
        return tokenizer(example["sentence"], truncation=True)
    elif task in ["mrpc", "rte", "wnli", "ax"]:
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    elif task in ["qqp"]:
        return tokenizer(example["question1"], example["question2"], truncation=True)
    elif task in ["qnli"]:
        return tokenizer(example["question"], example["sentence"], truncation=True)
    elif task in ["mnli"]:
        return tokenizer(example["premise"], example["hypothesis"], truncation=True)
    elif task in ["boolq"]:
        encoded_example = tokenizer(
            example["question"], example["passage"], truncation=True
        )
        # Convert the labels to a list of integers
        encoded_example["labels"] = [int(answer) for answer in example["answer"]]
        return encoded_example
    elif task in ["piqa"]:
        return tokenizer(
            example["goal"], example["sol1"], example["sol2"], truncation=True
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
