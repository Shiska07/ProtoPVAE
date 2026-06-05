# optimizer config for different stages

warm_vae_optimizer_lrs = {'vae': 3e-3}

warm_protoL_optimizer_lrs = {'vae': 3e-3,
                             'proto_bnorm': 3e-3,
                             'prototype_vectors': 3e-3}

joint_optimizer_lrs = {'features': 1e-4,
                       'vae': 3e-3,
                       'proto_bnorm': 3e-3,
                       'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 5e-4

default_lr = 1e-4