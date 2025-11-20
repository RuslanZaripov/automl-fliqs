import argparse

SUPPORTED_GLUE_TASKS = ["cola", "sst2", "mrpc", "qqp", "qnli", "rte", "wnli", "mnli"]
QA_TASKS = ["boolq", "piqa"]
VISION_TASKS = ["cifar10", "cifar100", "imagenet-1k"]
ALL_TASKS = SUPPORTED_GLUE_TASKS + QA_TASKS + VISION_TASKS

LANGUAGE_MODEL_MAP = {
    "bert": "bert-base-uncased",
    "bert-tiny": "prajjwal1/bert-tiny",
    "bert-mini": "prajjwal1/bert-mini",
    "bert-small": "prajjwal1/bert-small",
    "bert-medium": "prajjwal1/bert-medium",
    "roberta": "roberta-base",
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
}

VISON_MODEL_MAP = {
    "resnet18": "microsoft/resnet-18",
    "resnet50": "microsoft/resnet-50",
    "efficientnet-b0": "google/efficientnet-b0",
    "efficientnet-b1": "google/efficientnet-b1",
    "efficientnet-b2": "google/efficientnet-b2",
    "efficientnet-b3": "google/efficientnet-b3",
    "efficientnet-b4": "google/efficientnet-b4",
    "efficientnet-b5": "google/efficientnet-b5",
    "efficientnet-b6": "google/efficientnet-b6",
    "efficientnet-b7": "google/efficientnet-b7",
    "mobilenetv2": "google/mobilenet_v2_1.0_224",
    "deit-tiny": "facebook/deit-tiny-patch16-224",
    "deit-small": "facebook/deit-small-patch16-224",
    "deit-base": "facebook/deit-base-patch16-224",
}

ALL_MODEL_MAP = {**LANGUAGE_MODEL_MAP, **VISON_MODEL_MAP}
ALL_MODELS = list(ALL_MODEL_MAP.keys())


def validate_task(task: str) -> str:
    """Validate the task is with the GLUE benchmarks."""
    if task.lower() not in ALL_TASKS:
        raise argparse.ArgumentTypeError(
            f"Invalid task '{task}'. Allowed tasks: {', '.join(ALL_TASKS)}"
        )
    return task.lower()


def validate_cost_function(value):
    """Validate the cost function"""
    valid_cost_functions = ["hard_exponential", "soft_exponential", "tunas_abs"]
    if value not in valid_cost_functions:
        raise argparse.ArgumentTypeError(
            f"Invalid cost function. Allowed values are {valid_cost_functions}"
        )
    return value


# Add a new function to validate the model type
def validate_model_type(value: str) -> str:
    if value not in ALL_MODEL_MAP:
        raise argparse.ArgumentTypeError(
            f"Invalid model type. Allowed values are {ALL_MODELS}"
        )
    return value


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument(
        "--model_type",
        type=validate_model_type,
        default="bert",
        help="The model type to use for training and evaluation",
    )
    parser.add_argument(
        "--cuda_visible_devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--task",
        type=validate_task,
        default="sst2",
        help="The GLUE task to train and evaluate on",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="no",
        help="The save strategy to use for checkpoints",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="output",
        help="Output directory for the results",
    )
    parser.add_argument(
        "--exp_name", type=str, default="default", help="Experiment name"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--fp16", action="store_true", help="Whether to use fp16")

    # Learning parameters
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluation steps")
    parser.add_argument("--log_steps", type=int, default=100, help="Logging steps")
    parser.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamw_torch", help="Optimizer to use"
    )

    # Quantization parameters
    parser.add_argument(
        "--weight_bits",
        type=int,
        default=0,
        help="Quantization weight bits with 0 bits for no quantization",
    )
    parser.add_argument(
        "--activation_bits",
        type=int,
        default=0,
        help="Quantization act bits with 0 bits for no quantization",
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="Whether to use per-channel quantization",
    )
    parser.add_argument(
        "--per_token", action="store_true", help="Whether to use per-token quantization"
    )
    parser.add_argument(
        "--quantize_self_attention",
        action="store_true",
        default=True,
        help="Whether to quantize the self-attention layer",
    )
    parser.add_argument(
        "--quantize_self_output",
        action="store_true",
        default=True,
        help="Whether to quantize the self-attention output layer",
    )
    parser.add_argument(
        "--quantize_output",
        action="store_true",
        default=True,
        help="Whether to quantize the output layer",
    )
    parser.add_argument(
        "--quantize_intermediate",
        action="store_true",
        default=True,
        help="Whether to quantize the intermediate layer",
    )
    parser.add_argument(
        "--quantize_pooler",
        action="store_true",
        default=False,
        help="Whether to quantize the pooler layer",
    )
    parser.add_argument(
        "--quantize_classifier",
        action="store_true",
        default=False,
        help="Whether to quantize the classifier layer",
    )
    parser.add_argument(
        "--act_range_percentile",
        type=float,
        default=0,
        help="Percentile for the activation range",
    )
    parser.add_argument(
        "--weight_range_percentile",
        type=float,
        default=0,
        help="Percentile for the weight range",
    )

    # Search parameters
    parser.add_argument(
        "--search", action="store_true", help="Whether to add quantization search"
    )
    parser.add_argument(
        "--search_space", type=str, default="4,8", help="The quantization search space"
    )
    parser.add_argument(
        "--single_decision",
        action="store_true",
        help="Whether to assign single decision",
    )
    parser.add_argument(
        "--block_search", action="store_true", help="Whether to use block-level search"
    )
    parser.add_argument(
        "--weight_only_search",
        action="store_true",
        help="Whether to search only the weight bits",
    )
    parser.add_argument(
        "--cost_function",
        type=validate_cost_function,
        default="tunas_abs",
        help="The cost function to use, options are [hard_exponent, soft_exponent, tunas_abs]",
    )
    parser.add_argument(
        "--cost_type",
        type=str,
        default="memory",
        help="The cost type to use, options are" " [memory, compute]",
    )
    parser.add_argument(
        "--delay_steps",
        type=int,
        default=2500,
        help="Number of steps to delay before starting reinforcement learning",
    )
    parser.add_argument(
        "--cost_beta",
        type=float,
        default=1.0,
        help="Beta value to balance the accuracy and cost",
    )
    parser.add_argument(
        "--cost_target",
        type=float,
        default=50,
        help="Target cost for the cost function in MiB for memory cost and GBOPs for compute cost",
    )
    parser.add_argument(
        "--rl_learning_rate",
        type=float,
        default=5e-3,
        help="Learning rate for the reinforcement learning optimizer",
    )
    parser.add_argument(
        "--max_entropy_beta",
        type=float,
        default=0.5,
        help="Maximum value for the entropy regularization coefficient (beta)",
    )
    parser.add_argument(
        "--entropy_schedule_type",
        type=str,
        default="cosine",
        help="Entropy schedule type",
    )
    parser.add_argument(
        "--use_subset",
        action="store_true",
        help="Whether to use a subset of the training data for faster experiments",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=100,
        help="Size of the subset to use if --use_subset is set",
    )

    return parser
