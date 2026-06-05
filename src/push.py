import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocess import preprocess_input_function
from utils import receptive_field
from configs.train_settings import class_specific, \
    save_prototype_class_identity
from utils.helpers import create_dir

from configs.proto_configs import num_prototypes, \
    proto_bound_boxes_filename_prefix, \
    input_height, prototype_activation_function, prototype_self_act_filename_prefix, \
    prototype_img_filename_prefix, epsilon, n_classes

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
'''

# change to loading checkpoints with hyperparameters
def save_pushed_model(model_state_dict,
                      prototype_model_saving_dir,
                      cycle_number,
                      name):

    model_dir = os.path.join(prototype_model_saving_dir, f"cycle_{cycle_number}")
    create_dir(model_dir)
    model_path = os.path.join(model_dir, name)
    torch.save(model_state_dict, model_path)
    print(f"Model state_dict {name} saved at "
          f"{prototype_model_saving_dir}.")


# push each prototype to the nearest patch in the training set
def push_prototypes(push_dataloader,
                    model,
                    cycle_number,
                    save_prototypes,
                    prototype_saving_dir,
                    prototype_shape
                    ):


    proto_cycle_dir = os.path.join(prototype_saving_dir,
                                        f"cycle{cycle_number}")
    create_dir(proto_cycle_dir)

    proto_cycle_dir_sec = os.path.join(prototype_saving_dir,
                                   f"cycle{cycle_number}_sec")
    create_dir(proto_cycle_dir_sec)

    if bool(save_prototypes):
        print(f"Prototype images will be saveed in PUSH CYCLE {cycle_number}.")

    '''
    saves the closest distance seen so far to track if the closest
    prototype has been found, initialized with infinity for each prototype
    '''
    global_min_proto_dist = np.full(num_prototypes, np.inf)
    global_min_proto_dist_sec = np.full(num_prototypes, np.inf)

    '''
    saves the latent space patch representation of training data that gives
    the current smallest distance
    '''
    global_min_fmap_patches = np.zeros(
        [num_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    global_min_fmap_patches_sec = np.zeros(
        [num_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
     proto_rf_boxes and proto_bound_boxes column:
     0: image index in the entire dataset
     1: height start index
     2: height end index
     3: width start index
     4: width end index
     5: (optional) img class identity/ true class
     6: prototype class identity
     '''

    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[num_prototypes, 7],
                                      fill_value=-1)
        proto_rf_boxes_sec = np.full(shape=[num_prototypes, 7],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[num_prototypes, 7],
                                         fill_value=-1)
        proto_bound_boxes_sec = np.full(shape=[num_prototypes, 7],
                                    fill_value=-1)
        proto_class_specificity = np.full(shape=[n_classes,],
                                              fill_value=-1)
        proto_class_specificity_sec = np.full(shape=[n_classes,],
                                    fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[num_prototypes, 5],
                                      fill_value=-1)
        proto_rf_boxes_sec = np.full(shape=[num_prototypes, 5],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[num_prototypes, 5],
                                         fill_value=-1)
        proto_bound_boxes_sec = np.full(shape=[num_prototypes, 5],
                                    fill_value=-1)
        proto_class_specificity = np.full(shape=[n_classes,],
                                          fill_value=-1)
        proto_class_specificity_sec = np.full(shape=[n_classes, ],
                                          fill_value=-1)

    search_batch_size = push_dataloader.batch_size
    num_batches = len(push_dataloader)


    for push_iter, (search_batch_input, search_y) in enumerate(
            push_dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        print(f"PUSHING PROTOTYPES batch {push_iter} out of {num_batches} batches.")

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   model,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   global_min_proto_dist_sec,
                                   global_min_fmap_patches_sec,
                                   proto_rf_boxes_sec,
                                   proto_bound_boxes_sec,
                                   search_y,
                                   proto_cycle_dir,
                                   proto_cycle_dir_sec,
                                   save_prototypes,
                                   prototype_shape,
                                   preprocess_input_function=preprocess_input_function)

    if proto_cycle_dir is not None and \
            proto_bound_boxes_filename_prefix \
            is not None:
        # save data corresponding to the receptive field
        np.save(os.path.join(proto_cycle_dir,
                             proto_bound_boxes_filename_prefix + "-receptive_field" + str(
                                 cycle_number) + ".npy"),
                proto_rf_boxes)

        # save data corresponding to the bounding boxes
        np.save(os.path.join(proto_cycle_dir,
                             proto_bound_boxes_filename_prefix + str(
                                 cycle_number) + ".npy"),
                proto_bound_boxes)

        for i in range(n_classes):
            mask = proto_rf_boxes[:, 6] == i
            proto_class_specificity[i] = np.logical_and((proto_rf_boxes[:, 5] ==
                                    proto_rf_boxes[:, 6]),
                           mask).sum()

        np.save(os.path.join(proto_cycle_dir,
                             proto_bound_boxes_filename_prefix +
                             'class_specificity' + str(
                                 cycle_number) + ".npy"),
                proto_class_specificity)

    if proto_cycle_dir_sec is not None and \
            proto_bound_boxes_filename_prefix \
            is not None:
        # save data corresponding to the receptive field
        np.save(os.path.join(proto_cycle_dir_sec,
                             proto_bound_boxes_filename_prefix + "-receptive_field" + str(
                                 cycle_number) + ".npy"),
                proto_rf_boxes_sec)

        # save data corresponding to the bounding boxes
        np.save(os.path.join(proto_cycle_dir_sec,
                             proto_bound_boxes_filename_prefix + str(
                                 cycle_number) + ".npy"),
                proto_bound_boxes_sec)

        for i in range(n_classes):
            mask = proto_rf_boxes_sec[:, 6] == i
            proto_class_specificity_sec[i] = np.logical_and((proto_rf_boxes_sec[:,
                                                           5] ==
                                    proto_rf_boxes_sec[:, 6]),
                           mask).sum()
        np.save(os.path.join(proto_cycle_dir_sec,
                             proto_bound_boxes_filename_prefix +
                             'class_specificity_sec' + str(
                                 cycle_number) + ".npy"),
                proto_bound_boxes_sec)


    print(f"\tExecuting push ...cycle {cycle_number}")
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    model.prototype_block.prototype_vectors.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32))


def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               model,
                               global_min_proto_dist,
                               global_min_fmap_patches,
                               proto_rf_boxes,
                               proto_bound_boxes,
                               global_min_proto_dist_sec,
                               global_min_fmap_patches_sec,
                               proto_rf_boxes_sec,
                               proto_bound_boxes_sec,
                               search_y,
                               proto_cycle_dir,
                               proto_cycle_dir_sec,
                               save_prototypes,
                               prototype_shape,
                               preprocess_input_function=None,
                               prototype_activation_function_in_numpy=None,
                               prototype_layer_stride=1):



    # preprocess batch if necessary
    if preprocess_input_function is not None:
        print("preprocessing input for pushing ...")
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input


    '''
    Compute forward upto the bottleneck to get latent space representation
    and results from _l2_convolution
    '''
    model.cuda()
    model.eval()
    with torch.no_grad():

        # send batch to gpu
        search_batch = search_batch.cuda()

        # using mean_distances for push
        protoL_input_torch, proto_dist_torch = model.push_forward(
            search_batch)

    # make sure values are between 0 and 1
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(n_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            '''
            dinctionary containing indices of images belonging to each class in a
            list.
            class label is the key and the corresponding list is the value.
            '''
            class_to_img_index_dict[img_label].append(img_index)

    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(num_prototypes):
        # if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(
                model.prototype_block.prototype_class_identity[
                    j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue

            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,
                           j, :, :]

            # for secondary prototypes it is not class specific i.e. dist w.r.t
            # entire batch
            proto_dist_j_sec = proto_dist_[:, j, :, :]
        else:
            # if it is not class specific, then we will search through
            # every example
            '''
            Get min_distances values for the batch with the jth prototype but only with images
            belonging to prototype j's class identity
            proto_dist_.shape = (batch_size, num_prototypes, 7, 7)
            proto_dist_j = (n, 7, 7) where n = number of images beloging to
            prototype j's class identity.
            '''
            proto_dist_j = proto_dist_[:, j, :, :]
            proto_dist_j_sec = proto_dist_[:, j, :, :]

        '''
        Returns the minimum value in the entire distance array i.e.
        distance with
        the closest training patch
        '''

        # PRIMARY PROTOTYPES
        batch_min_proto_dist_j = np.amin(proto_dist_j)

        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            '''
            If the distance found is the smalles so far, find the 3D index at
            which the value exists
            '''
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmin_proto_dist_j[0] = \
                    class_to_img_index_dict[target_class][
                        batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]

            '''
            Get index information for extracting closest training patch.
            '''
            fmap_height_start_index = batch_argmin_proto_dist_j[
                                          1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[
                                         2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            '''
            Extract patch from the closeset training image.
            '''
            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                     :,
                                     fmap_height_start_index:fmap_height_end_index,
                                     fmap_width_start_index:fmap_width_end_index]

            # all index info of the closest training patch
            global_min_proto_dist[j] = batch_min_proto_dist_j

            # value of the closest traning patch
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            '''
            This part uses the receptive field information to generate
            visualization in the pixel space.
            '''
            # get the receptive field boundary of the image patch
            # that generates the representation
            layer_filter_sizes, layer_strides, layer_paddings = \
                model.conv_info()

            '''
            Compute receptive field at prototype layer
            '''
            protoL_rf_info = receptive_field.compute_proto_layer_rf_info_v2(
                input_height,
                layer_filter_sizes=layer_filter_sizes,
                layer_strides=layer_strides,
                layer_paddings=layer_paddings,
                prototype_kernel_size=prototype_shape[2])

            '''
            Using the network's receptive field, find the corresponding
            spatial indices for cropping in image space. [y1, y1, x1, x2]
            '''
            rf_prototype_j = receptive_field.compute_rf_prototype(input_height,
                                                                  batch_argmin_proto_dist_j,
                                                                  protoL_rf_info)

            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()

            '''
            original shape is (channels, height, width)
            transpose to (height, width, channels)
            '''
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]

            # crop out the receptive field covered by th prototype in the
            # original image
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                       rf_prototype_j[3]:rf_prototype_j[4], :]

            '''
            save the prototype receptive field information
            proto_rf_boxes and proto_bound_boxes column:
            0: image index in the entire dataset
            1: height start index
            2: height end index
            3: width start index
            4: width end index
            5: (optional) img class identity/ true class
            6: prototype class identity
            '''
            proto_rf_boxes[j, 0] = rf_prototype_j[
                                            0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 7 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()
                proto_rf_boxes[j, 6] = target_class

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]

            # apply activation function to distance for visualization
            if prototype_activation_function == "log":
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + epsilon))
            elif prototype_activation_function == "linear":
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(
                    proto_dist_img_j)

            # upsample the activation map
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(
                original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = receptive_field.find_high_activation_crop(
                upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                          proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 7 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()
                proto_bound_boxes[j, 6] = target_class

            if proto_cycle_dir is not None and bool(save_prototypes):

                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation

                    np.save(os.path.join(proto_cycle_dir,
                                         prototype_self_act_filename_prefix + str(
                                             j) + ".npy"),
                            proto_act_img_j)

                if prototype_img_filename_prefix is not None:

                    '''
                    1. SAVE complete original image of prototype
                    '''
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(proto_cycle_dir,
                                            prototype_img_filename_prefix + '-original' + str(
                                                j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(
                        upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(
                        rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(
                        np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap

                    '''
                    2. SAVE complete original image overlayed with activation map
                    '''

                    plt.imsave(os.path.join(proto_cycle_dir,
                                            prototype_img_filename_prefix + "-original_with_self_act" + str(
                                                j) + ".png"),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)

                    '''
                    3. SAVE part of original image corresponding to the prototype's
                    receptive field
                    '''
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[
                        1] != original_img_size:
                        plt.imsave(os.path.join(proto_cycle_dir,
                                                prototype_img_filename_prefix + "-receptive_field" + str(
                                                    j) + ".png"),
                                   rf_img_j,
                                   cmap="gray")

                        '''
                        4. SAVE image corresponding to the prototype's
                        receptive field overlayed with activation map
                        '''
                        overlayed_rf_img_j = overlayed_original_img_j[
                                             rf_prototype_j[1]:rf_prototype_j[2],
                                             rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(proto_cycle_dir,
                                                prototype_img_filename_prefix + "-receptive_field_with_self_act" \
                                                + str(j) + ".png"),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)

                    # save the prototype image (highly activated region of the whole image)
                    '''
                    5. SAVE part of the original image corresponding to the highly
                    activated region.
                    '''
                    plt.imsave(os.path.join(proto_cycle_dir,
                                            prototype_img_filename_prefix + str(
                                                j) + ".png"),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)


        # FOR SECONDARY PROTOTYPES
        batch_min_proto_dist_j_sec = np.amin(proto_dist_j_sec)
        if batch_min_proto_dist_j_sec < global_min_proto_dist_sec[j]:
            '''
            If the distance found is the smallest so far, find the 3D index at
            which the value exists
            '''
            batch_argmin_proto_dist_j_sec = \
                list(np.unravel_index(np.argmin(proto_dist_j_sec, axis=None),
                                      proto_dist_j_sec.shape))

            # retrieve the corresponding feature map patch
            img_index_in_batch_sec = batch_argmin_proto_dist_j_sec[0]

            '''
            Get index information for extracting closest training patch.
            '''
            fmap_height_start_index_sec = batch_argmin_proto_dist_j_sec[
                                          1] * prototype_layer_stride
            fmap_height_end_index_sec = fmap_height_start_index_sec + proto_h
            fmap_width_start_index_sec = batch_argmin_proto_dist_j_sec[
                                         2] * prototype_layer_stride
            fmap_width_end_index_sec = fmap_width_start_index_sec + proto_w

            '''
            Extract patch from the closeset training image.
            '''
            batch_min_fmap_patch_j_sec = protoL_input_[img_index_in_batch_sec,
                                     :,
                                     fmap_height_start_index_sec:fmap_height_end_index_sec,
                                     fmap_width_start_index_sec:fmap_width_end_index_sec]

            # all index info of the closest training patch
            global_min_proto_dist_sec[j] = batch_min_proto_dist_j_sec

            # value of the closest traning patch
            global_min_fmap_patches_sec[j] = batch_min_fmap_patch_j_sec

            '''
            This part uses the receptive field information to generate
            visualization in the pixel space.
            '''
            # get the receptive field boundary of the image patch
            # that generates the representation
            layer_filter_sizes, layer_strides, layer_paddings = \
                model.conv_info()

            '''
            Compute receptive field at prototype layer
            '''
            protoL_rf_info_sec = receptive_field.compute_proto_layer_rf_info_v2(
                input_height,
                layer_filter_sizes=layer_filter_sizes,
                layer_strides=layer_strides,
                layer_paddings=layer_paddings,
                prototype_kernel_size=prototype_shape[2])

            '''
            Using the network's receptive field, find the corresponding
            spatial indices for cropping in image space. [y1, y1, x1, x2]
            '''
            rf_prototype_j_sec = receptive_field.compute_rf_prototype(input_height,
                                                                  batch_argmin_proto_dist_j_sec,
                                                                  protoL_rf_info_sec)

            # get the whole image
            original_img_j_sec = search_batch_input[rf_prototype_j_sec[0]]
            original_img_j_sec = original_img_j_sec.numpy()

            '''
            original shape is (channels, height, width)
            transpose to (height, width, channels)
            '''
            original_img_j_sec = np.transpose(original_img_j_sec, (1, 2, 0))
            original_img_size_sec = original_img_j_sec.shape[0]

            # crop out the receptive field covered by th prototype in the
            # original image
            rf_img_j_sec = original_img_j_sec[rf_prototype_j_sec[1]:rf_prototype_j_sec[2],
                       rf_prototype_j_sec[3]:rf_prototype_j_sec[4], :]

            '''
            save the prototype receptive field information
            proto_rf_boxes and proto_bound_boxes column:
            0: image index in the entire dataset
            1: height start index
            2: height end index
            3: width start index
            4: width end index
            5: (optional) class identity/ true class
            6: prototype class identity
            '''
            proto_rf_boxes_sec[j, 0] = rf_prototype_j_sec[
                                            0] + start_index_of_search_batch
            proto_rf_boxes_sec[j, 1] = rf_prototype_j_sec[1]
            proto_rf_boxes_sec[j, 2] = rf_prototype_j_sec[2]
            proto_rf_boxes_sec[j, 3] = rf_prototype_j_sec[3]
            proto_rf_boxes_sec[j, 4] = rf_prototype_j_sec[4]
            if proto_rf_boxes_sec.shape[1] == 7 and search_y is not None:
                proto_rf_boxes_sec[j, 5] = search_y[rf_prototype_j_sec[0]].item()
                proto_rf_boxes_sec[j, 6] = target_class

            # find the highly activated region of the original image
            proto_dist_img_j_sec = proto_dist_[img_index_in_batch_sec, j, :, :]

            # apply activation function to distance for visualization
            if prototype_activation_function == "log":
                proto_act_img_j_sec = np.log((proto_dist_img_j_sec + 1) / (proto_dist_img_j_sec + epsilon))
            elif prototype_activation_function == "linear":
                proto_act_img_j_sec = max_dist - proto_dist_img_j_sec
            else:
                proto_act_img_j_sec = prototype_activation_function_in_numpy(
                    proto_dist_img_j_sec)

            # upsample the activation map
            upsampled_act_img_j_sec = cv2.resize(proto_act_img_j_sec, dsize=(
                original_img_size_sec, original_img_size_sec),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j_sec = receptive_field.find_high_activation_crop(
                upsampled_act_img_j_sec)
            # crop out the image patch with high activation as prototype image
            proto_img_j_sec = original_img_j_sec[proto_bound_j_sec[0]:proto_bound_j_sec[1],
                          proto_bound_j_sec[2]:proto_bound_j_sec[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes_sec[j, 0] = proto_rf_boxes_sec[j, 0]
            proto_bound_boxes_sec[j, 1] = proto_bound_j_sec[0]
            proto_bound_boxes_sec[j, 2] = proto_bound_j_sec[1]
            proto_bound_boxes_sec[j, 3] = proto_bound_j_sec[2]
            proto_bound_boxes_sec[j, 4] = proto_bound_j_sec[3]
            if proto_bound_boxes_sec.shape[1] == 7 and search_y is not None:
                proto_bound_boxes_sec[j, 5] = search_y[rf_prototype_j_sec[0]].item()
                proto_bound_boxes_sec[j, 6] = target_class

            if proto_cycle_dir_sec is not None and bool(save_prototypes):

                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation

                    np.save(os.path.join(proto_cycle_dir_sec,
                                         prototype_self_act_filename_prefix + str(
                                             j) + ".npy"),
                            proto_act_img_j_sec)

                if prototype_img_filename_prefix is not None:

                    '''
                    1. SAVE complete original image of prototype
                    '''
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(proto_cycle_dir_sec,
                                            prototype_img_filename_prefix + '-original' + str(
                                                j) + '.png'),
                               original_img_j_sec,
                               vmin=0.0,
                               vmax=1.0)

                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j_sec = upsampled_act_img_j_sec - np.amin(
                        upsampled_act_img_j_sec)
                    rescaled_act_img_j_sec = rescaled_act_img_j_sec / np.amax(
                        rescaled_act_img_j_sec)
                    heatmap = cv2.applyColorMap(
                        np.uint8(255 * rescaled_act_img_j_sec), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    overlayed_original_img_j_sec = 0.5 * original_img_j_sec + 0.3 * heatmap

                    '''
                    2. SAVE complete original image overlayed with activation map
                    '''

                    plt.imsave(os.path.join(proto_cycle_dir_sec,
                                            prototype_img_filename_prefix + "-original_with_self_act" + str(
                                                j) + ".png"),
                               overlayed_original_img_j_sec,
                               vmin=0.0,
                               vmax=1.0)

                    '''
                    3. SAVE part of original image corresponding to the prototype's
                    receptive field
                    '''
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j_sec.shape[0] != original_img_size_sec or rf_img_j_sec.shape[
                        1] != original_img_size_sec:
                        plt.imsave(os.path.join(proto_cycle_dir_sec,
                                                prototype_img_filename_prefix + "-receptive_field" + str(
                                                    j) + ".png"),
                                   rf_img_j_sec,
                                   cmap="gray")

                        '''
                        4. SAVE image corresponding to the prototype's
                        receptive field overlayed with activation map
                        '''
                        overlayed_rf_img_j_sec = overlayed_original_img_j_sec[
                                             rf_prototype_j_sec[1]:rf_prototype_j_sec[2],
                                             rf_prototype_j_sec[3]:rf_prototype_j_sec[4]]
                        plt.imsave(os.path.join(proto_cycle_dir_sec,
                                                prototype_img_filename_prefix + "-receptive_field_with_self_act" \
                                                + str(j) + ".png"),
                                   overlayed_rf_img_j_sec,
                                   vmin=0.0,
                                   vmax=1.0)

                    # save the prototype image (highly activated region of the whole image)
                    '''
                    5. SAVE part of the original image corresponding to the highly
                    activated region.
                    '''
                    plt.imsave(os.path.join(proto_cycle_dir_sec,
                                            prototype_img_filename_prefix + str(
                                                j) + ".png"),
                               proto_img_j_sec,
                               vmin=0.0,
                               vmax=1.0)


    if class_specific:
        del class_to_img_index_dict

