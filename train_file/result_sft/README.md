---
library_name: peft
license: other
base_model: /root/autodl-tmp/Hugging-Face/models/DeepSeek-R1-Distill-Qwen-7B
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft-01
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft-01

This model is a fine-tuned version of [/root/autodl-tmp/Hugging-Face/models/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co//root/autodl-tmp/Hugging-Face/models/DeepSeek-R1-Distill-Qwen-7B) on the medical_train_zh_top100k dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 3
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 5
- total_train_batch_size: 30
- total_eval_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 2.5

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.52.1
- Pytorch 2.8.0.dev20250605+cu128
- Datasets 3.6.0
- Tokenizers 0.21.1