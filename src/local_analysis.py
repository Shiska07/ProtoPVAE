import os
import copy
import argparse

import cv2
import numpy as np
from PIL import Image
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utils.log import create_logger
from utils.receptive_field import find_high_activation_crop
from utils.preprocess import mean, std, undo_preprocess_input_function
from utils.helpers import create_dir, load_parameters

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
'''


'''
Sample run command:
python local_analysis.py \
    --gpuid 0 \
    --data_dir ./datasets \
    --session_name session1 \
    --model_dir ./session1/final_models \
    --cycle_number 0 \
    --model_name last_layer_cycle_0.pth \
    --proto_info_dir ./session1/prototypes/cycle0 \
    --hparams_file ./session1/push_models/hparams.json \
    --img_dir ./local_analysis/189.Red_bellied_Woodpecker \
    --img_name Red_Bellied_Woodpecker_0007_182242.jpg \
    --img_class 189
    
'''

parser = argparse.ArgumentParser()

parser.add_argument('--gpuid', type=str, default='0',
    help='GPU ID(s) to use for training/inference (e.g. "0" or "0,1")')

parser.add_argument('--data_dir', type=str, default=os.path.join('.', 'datasets'),
    help='path to the root dataset directory')

parser.add_argument('--session_name',  type=str, default='session1')

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

parser.add_argument('--img_dir', type=str,
    help='path to the directory containing the test image')

parser.add_argument('--img_name', type=str,
    help='filename of the test image to analyze')

parser.add_argument('--img_class', type=int, default=-1,
    help='ground-truth class index of the test image (-1 if unknown)')

parser.add_argument('--use_sample', type=bool, default=True,
    help='whether to use sample or mean during inference')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

##### MODEL AND DATA LOADING
# specify the test image to be analyzed
test_image_dir = args.img_dir #'./local_analysis/Painted_Bunting_Class15_0081/'
test_image_name = args.img_name #'Painted_Bunting_0081_15230.jpg'
test_image_label = args.img_class #15
hparams = load_parameters(args.hparams_file)

test_image_path = os.path.join(test_image_dir, test_image_name)

# load the model
# check_test_accu = False

load_model_dir = args.model_dir #'.session1/final_models'
cycle_number = args.cycle_number
load_model_name = args.model_name #'model.pth'
prototype_info_dir = args.proto_info_dir
model_base_architecture = hparams['base_architecture']
experiment_session = args.session_name

save_analysis_path = os.path.join(test_image_dir, model_base_architecture,
                                  experiment_session, load_model_name,
                                  test_image_name.split('.')[0],
                                  f'use_samp_{int(args.use_sample)}')
create_dir(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + args.session_name)
log('use samples: ' + str(args.use_sample))

ppvae_model = torch.load(load_model_path)
ppnet_multi = torch.nn.DataParallel(ppvae_model)

img_size = ppnet_multi.module.img_size
prototype_shape = ppvae_model.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# load the test data and check test accuracy
test_dir = os.path.join(args.data_dir, "cub200_cropped",
                                     "test_cropped")

##### SANITY CHECK
# confirm prototype class identity
# save prototypes in original images
load_img_dir = os.path.join(prototype_info_dir, 'cycle'+str(cycle_number))
prototype_info = np.load(os.path.join(prototype_info_dir, 'cycle'+str(cycle_number), 'bb'+str(cycle_number)+'.npy'))

prototype_img_identity = prototype_info[:, -1]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppvae_model.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppvae_model.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')

##### HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(fname, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'prototype-img'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)
    
def save_prototype_self_activation(fname,  index):
    p_img = plt.imread(os.path.join(load_img_dir,
                                    'prototype-img-original_with_self_act'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)

def save_prototype_original_img_with_bbox(fname, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir,  'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    plt.imsave(fname, p_img_rgb)

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    #plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)

# load the test image and forward it through the network
preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   normalize
])

img_pil = Image.open(test_image_path)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))

images_test = img_variable.cuda()
labels_test = torch.tensor([test_image_label])

# forward
logits, distances, min_distances = ppnet_multi(images_test, use_samp=args.use_sample)

# we use sample_distances for local anaysis as we need to sample multiple times to
prototype_activations = ppvae_model.prototype_block.distance_2_similarity(
    min_distances)
prototype_activation_patterns = ppvae_model.prototype_block.distance_2_similarity(
    distances)
if ppvae_model.prototype_activation_function == 'linear':
    prototype_activations = prototype_activations + max_dist
    prototype_activation_patterns = prototype_activation_patterns + max_dist

tables = []

# sort the activation values to idemtify top-k activates prototypes
for i in range(logits.size(0)):
    tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[0].item()))
    log(str(i) + ' ' + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
log('Predicted: ' + str(predicted_cls))
log('Actual: ' + str(correct_cls + 1))
original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     images_test, idx)

##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
create_dir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

log('Most activated 10 prototypes of this image:')
array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
for i in range(1,11):
    log('top {0} activated prototype for this image:'.format(i))
    save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_prototype.png' % i),
                    sorted_indices_act[-i].item())

    # save prototype image and activation mao for top ith prototype
    save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          index=sorted_indices_act[-i].item(),
                                          bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                          bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                          bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                          bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                          color=(0, 255, 255))
    save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                'top-%d_activated_prototype_self_act.png' % i),
                                    sorted_indices_act[-i].item())
    log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
    log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
    if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
        log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
    log('activation value (similarity score): {0}'.format(array_act[-i]))
    log('last layer connection with predicted class: {0}'.format(ppvae_model.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
    
    activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
    upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                              interpolation=cv2.INTER_CUBIC)
    
    # show the most highly activated patch of the image by this prototype
    high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
    high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                  high_act_patch_indices[2]:high_act_patch_indices[3], :]
    log('most highly activated patch of the chosen image by this prototype:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_by_top-%d_prototype.png' % i),
               high_act_patch)
    log('most highly activated patch by this prototype shown in the original image:')
    imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                     img_rgb=original_img,
                     bbox_height_start=high_act_patch_indices[0],
                     bbox_height_end=high_act_patch_indices[1],
                     bbox_width_start=high_act_patch_indices[2],
                     bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
    
    # show the image overlayed with prototype activation map
    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
    rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    overlayed_img = 0.5 * original_img + 0.3 * heatmap
    log('prototype activation map of the chosen image:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'prototype_activation_map_by_top-%d_prototype.png' % i),
               overlayed_img)
    log('--------------------------------------------------------------')

##### PROTOTYPES FROM TOP-k CLASSES
k = 5
log('Prototypes from top-%d classes:' % k)
topk_logits, topk_classes = torch.topk(logits[idx], k=k)
for i,c in enumerate(topk_classes.detach().cpu().numpy()):
    create_dir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

    log('top %d predicted class: %d' % (i+1, c))
    log('logit of the class: %f' % topk_logits[i])
    class_prototype_indices = np.nonzero(
        ppvae_model.prototype_block.prototype_class_identity.detach().cpu().numpy(

        )[:, c])[0]
    class_prototype_activations = prototype_activations[idx][class_prototype_indices]
    _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

    prototype_cnt = 1
    for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        prototype_index = class_prototype_indices[j]
        save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'top-%d_activated_prototype.png' % prototype_cnt),
                        prototype_index)
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                                 'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                              index=prototype_index,
                                              bbox_height_start=prototype_info[prototype_index][1],
                                              bbox_height_end=prototype_info[prototype_index][2],
                                              bbox_width_start=prototype_info[prototype_index][3],
                                              bbox_width_end=prototype_info[prototype_index][4],
                                              color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                    'top-%d_activated_prototype_self_act.png' % prototype_cnt), prototype_index)
        log('prototype index: {0}'.format(prototype_index))
        log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
        if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
            log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
        log('last layer connection: {0}'.format(ppvae_model.last_layer.weight[c][prototype_index]))
        
        activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                  interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                      high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                   high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                         img_rgb=original_img,
                         bbox_height_start=high_act_patch_indices[0],
                         bbox_height_end=high_act_patch_indices[1],
                         bbox_width_start=high_act_patch_indices[2],
                         bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                   overlayed_img)
        log('--------------------------------------------------------------')
        prototype_cnt += 1
    log('***************************************************************')

if predicted_cls == correct_cls:
    log('Prediction is correct.')
else:
    log('Prediction is wrong.')

logclose()

