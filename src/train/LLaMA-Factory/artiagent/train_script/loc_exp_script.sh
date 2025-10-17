cd /home/jhpark/image-artifacts/src/train/LLaMA-Factory/

CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train artiagent/train_json/pointwise_vqa.yaml


cd /home/jhpark/image-artifacts/src/eval_vlm/
CUDA_VISIBLE_DEVICES=0 python eval_multi_task_vqa_batch.py --exp-dir /data2/jhpark/image-artifacts/vlm/saves/qwen2_5vl-7b/pointwise_vqa --dataset loki --dataset-path /data2/jhpark/image-artifacts/data/eval/loki --batch-size 16