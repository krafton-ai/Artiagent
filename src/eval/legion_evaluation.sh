#!/bin/bash

# WSOL Evaluation (Localization 1)
python wsol_eval.py --model 'legion' --dataset 'synthscars' --type 'localization'
python wsol_eval.py --model 'legion' --dataset 'synartifact' --type 'localization'
python wsol_eval.py --model 'legion' --dataset 'loki' --type 'localization'
python wsol_eval.py --model 'legion' --dataset 'richhf' --type 'localization'

# LEGION-like Evaluation (Localization 2)
python legion_eval.py --model 'legion' --dataset 'synthscars' --type 'localization'
python legion_eval.py --model 'legion' --dataset 'synartifact' --type 'localization'
python legion_eval.py --model 'legion' --dataset 'loki' --type 'localization'
python legion_eval.py --model 'legion' --dataset 'richhf' --type 'localization'

# Bbox-map Evaluation (Localization 3)
python eval.py --model 'legion' --dataset 'synthscars' --type 'localization'
python eval.py --model 'legion' --dataset 'synartifact' --type 'localization'
python eval.py --model 'legion' --dataset 'loki' --type 'localization'
python eval.py --model 'legion' --dataset 'richhf' --type 'localization'

# Explanation
python eval.py --model 'legion' --dataset 'synthscars' --type 'explanation'
python eval.py --model 'legion' --dataset 'loki' --type 'explanation'