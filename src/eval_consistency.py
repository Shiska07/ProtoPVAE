import os
import torch
import argparse
from utils.eval_interpretability import evaluate_consistency
from utils.helpers import create_dir

'''
Adapted from EvalProtoPNet (Huang et al. 2023).
Original: https://github.com/hqhQAQ/EvalProtoPNet
'''

'''
Sample run command:
python src/eval_consistency.py --model_dir ./session1/final_models \
                --model_name last_layer_cycle_0.pth \
                --proto_info_dir ./session1/prototypes
'''
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--data_set', default='CUB2011', type=str)

# make sure to add the original CUB_200_2011 folder with metadata to datasets/
parser.add_argument('--data_path', type=str, default='datasets/cub200_cropped/')
parser.add_argument('--nb_classes', type=int, default=200)
parser.add_argument('--test_batch_size', type=int, default=30)
parser.add_argument('--use_sample', type=int, default=0)

# Model
parser.add_argument('--model_dir',  type=str)
parser.add_argument('--model_name',  type=str)
parser.add_argument('--proto_info_dir',  type=str)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
load_model_dir = args.model_dir
load_model_name = args.model_name
prototype_info_dir = args.proto_info_dir
load_model_path = os.path.join(load_model_dir, load_model_name)

# load the model
print('load model from ' + load_model_path)
ppvae_model = torch.load(load_model_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
ppvae_model = ppvae_model.cuda()

consistency_score = evaluate_consistency(ppvae_model, args)
scores_dir = os.path.join(args.model_dir, 'scores')
create_dir(scores_dir)
fname = f"consistency_score_use_samp_{args.use_sample}_{consistency_score:0.7f}.txt"
file_path = os.path.join(scores_dir, fname)
with open(file_path, 'w') as file:
    file.write(f'consistency_score : {consistency_score:0.7f}')
print('Consistency Score : {:.2f}%'.format(consistency_score))