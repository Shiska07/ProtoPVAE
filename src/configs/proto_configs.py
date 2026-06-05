# configuration specific to prototypes
input_height = 224
input_channels = 3
n_classes = 200
num_prototypes = 2000

# this is used to convert distance to similarity values
prototype_activation_function = 'log'
epsilon = 1e-4

weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'
