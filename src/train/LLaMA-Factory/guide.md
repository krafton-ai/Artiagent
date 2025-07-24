# Overview
Llama factory 를 통해 학습하는 방법에 대해 정리한다.
# Method
## 0. Installation
- 현재 skythought 에서 할 때는, 아래와 같은 흐름으로 Llama-Factory 를 설치하고 사용함.
```bash
mkdir skythought/train
cd skythought/train
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```
## 1. Dataset 준비
- `dataset_info.json` 에 우리가 원하는 dataset 을 정의해서 넣어둬야 함. 이 정보가 나중에 학습에 쓸 yaml 파일에 입력으로 들어감. (`dataset_info.json` 은 다음 경로에 있음: `LLaMA-Factory/data/dataset_info.json`)
- 아래의 부분들을 참고해서 `dataset_info.json` 에 적어서 넣어둬야 함.
```json
"dataset_name": {
  "hf_hub_url": "the name of the dataset repository on the Hugging Face hub. (if specified, ignore script_url, file_name and cloud_file_name)",
  "ms_hub_url": "the name of the dataset repository on the Model Scope hub. (if specified, ignore script_url, file_name and cloud_file_name)",
  "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore file_name and cloud_file_name)",
  "cloud_file_name": "the name of the dataset file in s3/gcs cloud storage. (if specified, ignore file_name)",
  "file_name": "the name of the dataset folder or dataset file in this directory. (required if above are not specified)",
  "formatting": "the format of the dataset. (optional, default: alpaca, can be chosen from {alpaca, sharegpt})",
  "ranking": "whether the dataset is a preference dataset or not. (default: False)",
  "subset": "the name of the subset. (optional, default: None)",
  "split": "the name of dataset split to be used. (optional, default: train)",
  "folder": "the name of the folder of the dataset repository on the Hugging Face hub. (optional, default: None)",
  "num_samples": "the number of samples in the dataset to be used. (optional, default: None)",
  "columns (optional)": {
    "prompt": "the column name in the dataset containing the prompts. (default: instruction)",
    "query": "the column name in the dataset containing the queries. (default: input)",
    "response": "the column name in the dataset containing the responses. (default: output)",
    "history": "the column name in the dataset containing the histories. (default: None)",
    "messages": "the column name in the dataset containing the messages. (default: conversations)",
    "system": "the column name in the dataset containing the system prompts. (default: None)",
    "tools": "the column name in the dataset containing the tool description. (default: None)",
    "images": "the column name in the dataset containing the image inputs. (default: None)",
    "videos": "the column name in the dataset containing the videos inputs. (default: None)",
    "audios": "the column name in the dataset containing the audios inputs. (default: None)",
    "chosen": "the column name in the dataset containing the chosen answers. (default: None)",
    "rejected": "the column name in the dataset containing the rejected answers. (default: None)",
    "kto_tag": "the column name in the dataset containing the kto tags. (default: None)"
  },
  "tags (optional, used for the sharegpt format)": {
    "role_tag": "the key in the message represents the identity. (default: from)",
    "content_tag": "the key in the message represents the content. (default: value)",
    "user_tag": "the value of the role_tag represents the user. (default: human)",
    "assistant_tag": "the value of the role_tag represents the assistant. (default: gpt)",
    "observation_tag": "the value of the role_tag represents the tool results. (default: observation)",
    "function_tag": "the value of the role_tag represents the function call. (default: function_call)",
    "system_tag": "the value of the role_tag represents the system prompt. (default: system, can override system column)"
  }
}
```
- 참고: https://github.com/hiyouga/LLaMA-Factory/tree/main/data#readme
- 예시: Openr1_Math 를 불러오는 방법 (`dataset_info.json` 안에 이와 같이 파일을 입력함. 이 부분은 huggingface 데이터 저장형태를 참고함.)
```json
...
"openr1_math": {
    "hf_hub_url": "open-r1/OpenR1-Math-220k",
    "subset": "all",
    "split": "train",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  },
...
```
## 2. Training config 준비
- yaml 파일을 아래와 같이 준비.
```
### Default
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
trust_remote_code: true
### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]
### dataset
dataset: openr1_math
template: qwen
cutoff_len: 32768
max_samples: 225129
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
### output
output_dir: saves/qwen2.5-7b/full/sft_openr1_math
logging_steps: 5
save_steps: 100
save_total_limit: 1
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: tensorboard  # choices: [none, wandb, tensorboard, swanlab, mlflow]
### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
### eval
# eval_dataset: alpaca_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
```
- 알아두며 좋을 사항: dataset은 여러개 입력할 땐 "," 로 이어서 쓰기.
## 3. Training script
- train/LLaMA-Factory 폴더 안에서 아래의 script 실행.
```bash
llamafactory-cli train examples/train_full/qwen2_5.yaml
```