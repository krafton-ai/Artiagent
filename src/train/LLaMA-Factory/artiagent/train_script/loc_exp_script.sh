cd /home/jovyan/image-artifacts/src/train/LLaMA-Factory/

CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train artiagent/train_json/vanilla_multi_vqa.yaml


cd /home/jovyan/image-artifacts/src/eval/
CUDA_VISIBLE_DEVICES=0 python multi_task_vqa_eval.py --exp-dir /home/jovyan/image-artifacts/vlm/saves/qwen2_5vl-7b/multi_vqa --dataset loki --dataset-path /home/jovyan/image-artifacts/data/eval/loki --batch-size 16
