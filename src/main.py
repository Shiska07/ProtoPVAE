import random
import numpy as np
from argparse import ArgumentParser

import torch
from datamodule import CUBDataModule
from train_and_test import TrainingPipiline
from configs.proto_configs import prototype_activation_function

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# if GPU ID is provided, use that else use all gpus
# if args.gpuid is not None:
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# Command-line interface.
#
# The standard reproduction workflow is a sequence of `--task` invocations
# run in order:
#
#   python main.py --task warm_vae       --session_name <name>
#   python main.py --task warm_proto     --session_name <name>
#   python main.py --task joint          --session_name <name> --cycle_number 0
#   python main.py --task push           --session_name <name> --cycle_number 0
#   python main.py --task last_layer     --session_name <name> --cycle_number 0
#   python main.py --task test_last_layer --session_name <name> --cycle_number 0
#
# Multi-cycle training (experimentation code) (joint -> push -> last_layer repeated) is supported
# by re-running joint/push/last_layer with incremented --cycle_number values.
# the reported numbers are single-cycle.

def main():

    parser = ArgumentParser()


    parser.add_argument("--data_dir",
                       type=str,
                       default="./datasets",
                       help="root data dir ")

    # Stage selector. Each invocation runs exactly one stage.
    parser.add_argument("--task",
                       type=str,
                       default="test_last_layer",
                       help="Task. Eg: warm_vae, warm_proto, joint, last_layer for training; push, pre_push_test, etc. for testing ")

    # CNN backbone for the feature extractor. Loaded with ImageNet-pretrained
    parser.add_argument("--base_architecture",
                        type=str,
                        default="resnet50",
                        help="Pre-trained architecture to use. Eg: resnet50, vgg19")

    # All stage outputs (logs, prototypes, push artefacts, final models) are
    # written under {root_dir}/{session_name}/.
    parser.add_argument("--root_dir",
                        type=str,
                        default="",
                        help="Directory containing data and source file")

    parser.add_argument("--session_name",
                        type=str,
                        default="session1",
                        help="Experiment session name. All logs/checkpoints will be saved at this location")

    # Cycle index for joint/push/last_layer stages. 0 = single-cycle, thesis run.
    # Ignored by warm_vae/warm_proto (the warm stages do not cycle).
    parser.add_argument("--cycle_number",
                        type=int,
                        default=0,
                        help="current cycle number if mode is 'joint' or 'last_layer'")

    parser.add_argument("--accelerator",
                        type=str,
                        default="gpu")

    parser.add_argument("--batch_size",
                        type=int,
                        default=64)

    parser.add_argument("--random_seed",
                        type=int,
                        default=42,
                        help="Seed value fpr pl.seed_everything()")

    # Whether the push stage should write prototype-img PNGs / bb npys to the
    # prototypes/ directory. Optional artefact for analysis tools; not required
    # for training itself.
    parser.add_argument("--save_prototypes",
                        nargs='?',
                        type=int,
                        default=1,
                        help="whether to save images of prototypes")


    args = parser.parse_args()
    params = vars(args)


    # set seed for multi GPU training
    set_seed(args.random_seed)

    data_dir = args.data_dir

    data_module = CUBDataModule(data_dir=data_dir,
                                batch_size=args.batch_size,
                                push_batch_size=args.batch_size,
                                num_workers=1)

    training_pipeline = TrainingPipiline(params,
                                         data_module)

    # Stage 1 - VAE warm-up. Trains encoder + decoder only.
    if args.task == "warm_vae":
        training_pipeline.fit_warm_vae()

    if args.task == "test_warm_vae":
        training_pipeline.test_model(args.task, "warm_vae")

    # Stage 2 - Prototype layer warm-up. Trains VAE + prototype_block.
    if args.task == "warm_proto":
        training_pipeline.fit_warm_proto()

    if args.task == "test_warm_proto":
        training_pipeline.test_model(args.task, "warm_proto")

    # Stage 3 - Joint training. Trains features + VAE + prototype_block.
    # The last layer is NOT trained here. For cycle_number > 0 the previous cycle's last_layer output
    # is loaded as the starting point.
    if args.task == "joint":
        training_pipeline.fit_joint(args.cycle_number)

    # Evaluates the post-joint, pre-push model.
    if args.task == "pre_push_test":
        training_pipeline.test_model(args.task, "joint", cycle_number=args.cycle_number)


    # Stage 4 - Prototype projection. Replaces each learned prototype with the
    # nearest training-image patch of its class, computed from the VAE posterior
    # MEAN.
    if args.task == "push":
        training_pipeline.push_cycle(args.cycle_number)

    # Evaluates the post-push model. Accuracy here is expected to drop vs.
    # pre_push_test; Stage 5 (last_layer) recovers it.
    if args.task == "post_push_test":
        training_pipeline.test_model(args.task, "push", cycle_number=args.cycle_number)


    # Stage 5 - Convex optimization of the bias-free last layer with L1 sparsity.
    if prototype_activation_function != "linear":
        if args.task == "last_layer":
            training_pipeline.fit_last_layer(args.cycle_number)

        if args.task == "test_last_layer":
            training_pipeline.test_model(args.task, "last_layer", cycle_number=args.cycle_number)

if __name__ == '__main__':
    main()