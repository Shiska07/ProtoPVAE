import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import find_nearest
from utils.helpers import create_dir, load_parameters
from utils.preprocess import preprocess_input_function

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
'''

# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
parser = argparse.ArgumentParser()

parser.add_argument('--gpuid', type=str, default='0',
    help='GPU ID(s) to use for training/inference (e.g. "0" or "0,1")')
parser.add_argument('--data_dir', type=str, default=os.path.join('.', 'datasets'),
    help='path to the root dataset directory')
parser.add_argument('--model_dir', type=str, default=os.path.join('.', 'session1', 'final_models'),
    help='path to the directory containing saved model .pth files')
parser.add_argument('--cycle_number', type=int, default=0,
    help='training cycle index to use (0-based)')
parser.add_argument('--model_name', type=str, default='last_layer_cycle_0.pth',
    help='filename of the model to load')
parser.add_argument(
    '--proto_info_dir',
    type=str,
    default=os.path.join('.', 'session1', 'prototypes', 'cycle0'),
    help='path to the directory containing prototype metadata for the given cycle'
)
parser.add_argument(
    '--hparams_file',
    type=str,
    default=os.path.join('.', 'session1', 'push_models', 'hparams.json'),
    help='path to hparams.json file'
)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
load_model_dir = args.model_dir
cycle_number = args.cycle_number
load_model_name = args.model_name
prototype_info_dir = args.proto_info_dir

hparams = load_parameters(args.hparams_file)
load_model_path = os.path.join(load_model_dir, load_model_name)

# load the model
print('load model from ' + load_model_path)
ppvae_model = torch.load(load_model_path)
ppvae_model = ppvae_model.cuda()

# loading state_dict from lightning checkpoint
ppvae_model_multi = torch.nn.DataParallel(ppvae_model)

img_size = ppvae_model_multi.module.img_size

# load the data
# must use unaugmented (original) dataset
batch_size = 100

train_dir = os.path.join(args.data_dir,
                                      "cub200_cropped", "train_cropped")
test_dir = os.path.join(args.data_dir, "cub200_cropped",
                                     "test_cropped")


'''
Do not normalize data here.
'''
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=False)

# test set: do not normalize
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=False)

root_dir_for_saving_train_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_train')
root_dir_for_saving_test_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_test')
create_dir(root_dir_for_saving_train_images)
create_dir(root_dir_for_saving_test_images)

# save prototypes in original images
load_img_dir = os.path.join(prototype_info_dir)
prototype_info = np.load(os.path.join(prototype_info_dir, 'bb'+str(cycle_number)+'.npy'))

def save_prototype_original_img_with_bbox(fname, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(fname, p_img_rgb)


for j in range(ppvae_model.num_prototypes):
    create_dir(os.path.join(root_dir_for_saving_train_images, str(j)))
    create_dir(os.path.join(root_dir_for_saving_test_images, str(j)))
    save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_train_images, str(j),
                                                             'prototype_in_original_pimg.png'),
                                          index=j,
                                          bbox_height_start=prototype_info[j][1],
                                          bbox_height_end=prototype_info[j][2],
                                          bbox_width_start=prototype_info[j][3],
                                          bbox_width_end=prototype_info[j][4],
                                          color=(0, 255, 255))
    save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_test_images, str(j),
                                                             'prototype_in_original_pimg.png'),
                                          index=j,
                                          bbox_height_start=prototype_info[j][1],
                                          bbox_height_end=prototype_info[j][2],
                                          bbox_width_start=prototype_info[j][3],
                                          bbox_width_end=prototype_info[j][4],
                                          color=(0, 255, 255))

k = 5

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
        model_parallel=ppvae_model_multi, # pytorch network with prototype_vectors
        k=k+1,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=print)

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
        model_parallel=ppvae_model_multi, # pytorch network with prototype_vectors
        k=k,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        log=print)
