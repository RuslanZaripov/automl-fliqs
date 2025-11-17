# Quantization Search

This repository contains the implementation of quantization architecture search for convolutional and transformer models for vision and language tasks. It accompanies the submission FLIQS: Mixed-Precision Floating-Point and Integer Quantization Search. The results in this submission were collected across two codebases: one proprietary that cannot be open-sourced and this open-source version. This code will be further improved during the submission period.

## Table of Contents
- [Introduction](#introduction)
- [File Structure](#file-structure)
- [Usage](#usage)
    - [Training Parameters](#training-parameters)
    - [Reinforcement Learning Parameters](#reinforcement-learning-parameters)
    - [Quantization Parameters](#quantization-parameters)
    - [Layers to Quantize Parameters](#layers-to-quantize-parameters)
    - [Miscellaneous Parameters](#miscellaneous-parameters)
- [Supported Models](#supported-models)
    - [Language Models](#language-models)
    - [Vision Models](#vision-models)

## Introduction
Quantization has become a mainstream compression technique for reducing model
size, computational requirements, and energy consumption for modern deep neural networks (DNNs).
With the improved numerical support in recent hardware,
including multiple variants of integer and floating point, mixed-precision quantization
has become necessary to achieve high-quality results with low model cost. 
Prior mixed-precision quantization methods have performed a post-training
quantization search, which compromises on accuracy, or a differentiable quantization search,
which leads to high memory usage from branching. Therefore, we propose the first one-shot mixed-precision 
quantization search that eliminates the need for retraining in both integer and low-precision floating point 
models. We evaluate our floating-point and integer quantization search (FLIQS) on multiple
convolutional networks and vision transformer models to discover Pareto-optimal
models. Our approach discovers models that improve upon uniform precision,
manual mixed-precision, and recent integer quantization search methods.

## File Structure
- `quant/`: Contains code related to quantization.
    - `quant_linear.py`: Code for quantized linear transformations.
    - `quant_module.py`: General code for creating quantized modules.
    - `quant_utils.py`: Utility functions for quantization.
    - `quant_conv2d.py`: Code for quantized convolution operations.

- `eval_parallel.py`: Code for running evaluations in parallel across GPU.

- `cost/`: Contains code related to cost computation.
    - `cost.py`: Code for cost functions that combine quality and performance

- `eval/`: Contains code related to evaluation.
    - `eval_run.py`: Main script for running evaluations for search (block, single-decision, layer-wise) and quantized and floating-point baselines.
    - `transformer_utils.py`: Utility functions for transformer models.
    - `eval_partitions.py`: Code for establishing the block partitions for the models
    - `eval_parse.py`: Script for parsing and validating evaluation runs.
    - `vision_utils.py`: Utility functions for vision models.

- `trainer/`: Contains code related to modifying the Huggingface transformer
    - `reinforce_trainer.py`: Main code for the reinforcement learning trainer

- `policy/`: Contains code related to policy models.
    - `simple_policy.py`: Code for a simple policy model.

## Usage
To run eval_parallel.py, use the command line to navigate to the directory containing the script and then execute it with the desired options. Below are some examples:

#### Example Usage
1. BERT model with block search:
```python eval_parallel.py --task mrpc --model bert --block_search --search_space 4,8 --num_train_epochs 3 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_root_dir output```

2. ResNet18 for vision task with a larger search space:
```python eval_parallel.py --task cifar10 --model resnet18 --search --search_space 2,4,8  --num_train_epochs 3 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_root_dir output```

3. GPT-2 with single_decision search on SST2:
```python eval_parallel.py --task sst2 --model gpt2 --single_decision --search  --num_train_epochs 3 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_root_dir output```

4. EfficientNetB0 baseline quantized model. Note that you need to setup your HF auth to use ImageNet:
```python eval_parallel.py --task imagenet-1k --model efficientnet-b0 --weight_bits 8 --activation_bits 8 --num_train_epochs 3 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_root_dir output```

In addition, you can specify many different paramters that modify the training hyper-parameters, search space, search granularity, etc.
#### Training Parameters
- `--num_train_epochs`: Set the number of training epochs.
- `--learning_rate`: Set the learning rate for the optimizer.
- `--train_batch_size`: Set the batch size for training.
- `--eval_batch_size`: Set the batch size for evaluation.
- `--gradient_accumulation_steps`: Set the number of steps for gradient accumulation.
- `--optimizer`: Set the optimizer to use.
- `--fp16`: Include this flag to use fp16.
- `--max_steps`: Set the maximum number of training steps.

#### Reinforcement Learning Parameters
- `--delay_steps`: Set the number of steps to delay before beginning to learn the RL policy.
- `--rl_learning_rate`: Set the learning rate for the reinforcement learning optimizer.
- `--cost_beta`: Set the initial value for the entropy regularization coefficient (beta).
- `--max_entropy_beta`: Set the maximum value for the entropy regularization coefficient (beta).
- `--entropy_schedule_type`: Set the entropy schedule type.
- `--cost_target`: Set the target cost for the cost function.

#### Quantization Parameters
- `--search`: Include this flag to enable quantization search.
- `--bits`: Set the number of quantization bits. Use 0 for no quantization.
- `--search_space`: Set the quantization search space, e.g., '4,8'.
- `--single_decision`: Include this flag to assign single decision.
- `--block_search`: Include this flag to enable block-level search.
- `--weight_only_search`: Include this flag to search only the weight bits.
- `--cost_function`: Set the cost function to use, options are ['hard_exponential', 'soft_exponential', 'tunas_abs'].
- `--cost_type`: Set the cost type to use, options are ['memory', 'compute'].
- `--weight_bits`: Set the quantization weight bits with 0 bits for no quantization.
- `--activation_bits`: Set the quantization act bits with 0 bits for no quantization.
- `--per_channel`: Include this flag to use per-channel quantization.
- `--per_token`: Include this flag to use per-token quantization.
- `--act_range_percentile`: Set the percentile for the activation range.
- `--weight_range_percentile`: Set the percentile for the weight range.

#### Layers to Quantize Parameters
- `--quantize_self_attention`: Include this flag to quantize the self-attention layer.
- `--quantize_self_output`: Include this flag to quantize the self-attention output layer.
- `--quantize_output`: Include this flag to quantize the output layer.
- `--quantize_intermediate`: Include this flag to quantize the intermediate layer.
- `--quantize_pooler`: Include this flag to quantize the pooler layer.
- `--quantize_classifier`: Include this flag to quantize the classifier layer.

#### Miscellaneous Parameters
- `--cuda_visible_devices`: Set the value for the CUDA_VISIBLE_DEVICES environment variable.
- `--eval_steps`: Set the number of steps for evaluation.
- `--log_steps`: Set the number of steps for logging.
- `--exp_name`: Set the name for the experiment.
- `--save_strategy`: Set the strategy to use for checkpoints.
- `--seed`: Set the random seed for reproducibility.
- `--gradient_checkpointing`: Include this flag to use gradient checkpointing.

## Supported Models
Below is a list of the models supported in this repository:

### Language Models

- `bert-tiny`, `bert-mini`, `bert-small`, `bert-medium`, `bert`: BERT (Bidirectional Encoder Representations from Transformers).

- `roberta`: RoBERTa (Robustly optimized BERT approach) modifies key hyperparameters and removes the next sentence prediction objective.

- `gpt2`, `gpt2-medium`, `gpt2-large`: GPT-2 is a transformer-based generative language model. The "medium" and "large" versions offer increased capacity.

### Vision Models

- `resnet18`, `resnet50`: ResNet models are a popular choice for image recognition tasks.

- `efficientnet-b0` to `efficientnet-b7`: EfficientNet is a series of models (b0 to b7) that use a new scaling method to uniformly scale all dimensions of depth/width/resolution for better performance.

- `mobilenetv2`: MobileNetV2 is designed for mobile visual recognition including classification, object detection, and semantic segmentation.

- `deit-tiny`, `deit-small`, `deit-base`: DeiT (Data-Efficient Image Transformers) are a series of transformer-based vision models that perform on par with convolutional networks on ImageNet while being data-efficient.

```
python eval_parallel.py \
    --task mrpc \
    --model_type bert \
    --block_search \
    --search_space 4,8 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --output_root_dir output

python eval_parallel.py --task cifar10 --model_type resnet18 --search --search_space 2,4,8 --num_train_epochs 3 --learning_rate 5e-5 --train_batch_size 32 --eval_batch_size 32 --output_root_dir output
```

- [FLIQS: One-Shot Mixed-Precision Floating-Point and Integer Quantization Search](https://openreview.net/forum?id=d69NqU8YmM)

- [FLIQS youtube](https://www.youtube.com/watch?v=WHxltlBGHiw)

- [EG-ENAS: Efficient and Generalizable Evolutionary Neural Architecture Search for Image Classification](https://openreview.net/forum?id=3YWElIrU8a#discussion)

- [quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)