import os
import torch
from model import PartProtoVAE
import pytorch_lightning as pl
from configs.strategy_configs import strategy
from configs.train_settings import use_validation
from configs.epoch_configs import max_epochs_dict
from custom_callbacks import getmodel_ckpt_callback
from push import push_prototypes
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.receptive_field import compute_proto_layer_rf_info_v2
from configs.vae_configs import kl_coeff, recon_coeff, latent_channels, sigmoid_shift, dropout_rate
from configs.proto_configs import input_height
from utils.helpers import create_dir

from model_components.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from model_components.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from model_components.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

# Trades a small amount of matmul precision for faster training on modern GPUs.
torch.set_float32_matmul_precision('medium')

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
'''

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}



class TrainingPipiline:
    def __init__(self,
                 params,
                 datamodule):

        self.params = params
        self.datamodule = datamodule

        # save dataloaders
        if use_validation:
            self.val_dataloader = self.datamodule.val_dataloader()
        else:
            self.val_dataloader = None

        self.train_dataloader = self.datamodule.train_dataloader()
        self.train_push_dataloader = self.datamodule.train_push_dataloader()
        self.test_dataloader = self.datamodule.test_dataloader()

        self.model = None
        self.trainer = None

        features = base_architecture_to_features[
            params['base_architecture']](pretrained=True)

        kernel_sizes, strides, paddings = \
            features.conv_info()

        proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=input_height,
                                                             layer_filter_sizes=kernel_sizes,
                                                             layer_strides=strides,
                                                             layer_paddings=paddings,
                                                             prototype_kernel_size=7)

        self.model_arc_dir = os.path.join(params['root_dir'],
                                          params['session_name'],
                                          'model_arc')
        self.root_logging_dir = os.path.join(params['root_dir'],
                                             params['session_name'],
                                             'logs')
        self.root_ckpt_dir = os.path.join(params['root_dir'],
                                          params['session_name'],
                                          'checkpoints')
        self.final_models_dir = os.path.join(params['root_dir'],
                                             params['session_name'],
                                             'final_models')
        self.root_results_dir = os.path.join(params['root_dir'],
                                          params['session_name'],
                                          'results')
        self.prototype_saving_dir = os.path.join(params['root_dir'],
                                                 params['session_name'],
                                                 'prototypes')
        self.prototype_model_saving_dir = os.path.join(params['root_dir'],
                                                       params['session_name'],
                                                       'push_models')

        self.model_params = {'base_architecture': params['base_architecture'],
                             'dropout_rate': dropout_rate,
                             'kl_coeff': kl_coeff,
                             'recon_coeff': recon_coeff,
                             'latent_channels': latent_channels,
                             'sigmoid_shift': sigmoid_shift,
                             'model_arc_dir': self.model_arc_dir,
                             'prototype_model_saving_dir': self.prototype_model_saving_dir,
                             'logging_dir': self.root_logging_dir,
                             'root_results_dir': self.root_results_dir,
                             'proto_layer_rf_info': proto_layer_rf_info,
                             'final_models_dir': self.final_models_dir}

    def save_entire_model(self, model_name):
        model_path = os.path.join(self.final_models_dir, model_name)
        if os.path.exists(self.final_models_dir):
            torch.save(self.model, model_path)
            print(f"Model saved at {model_path}.")
        else:
            create_dir(self.final_models_dir)
            torch.save(self.model, model_path)
            print(f"Model saved at {model_path}.")

    def load_entire_model(self, model_name):
        model_path = os.path.join(self.final_models_dir, model_name)
        if os.path.exists(model_path):
            self.model = torch.load(model_path)
            print(f"Model loaded from {model_path}.")
        else:
            print(f"Model could not be loaded as dir {model_path} does not exist.")

    def initialize_trainer(self, train_mode, callbacks, logger):
        """
        Construct a Lightning Trainer with stage-specific epoch count and the
        caller's callbacks plus an LR monitor. NOTE: limit_train_batches and
        limit_val_batches are currently set to 30 for fast iteration; remove
        these before a real reproduction run.
        """
        callbacks = list(callbacks) + [LearningRateMonitor(logging_interval="epoch")]

        self.trainer = pl.Trainer(
            accelerator=self.params['accelerator'],
            strategy=strategy["auto"],
            devices=1,
            logger=logger,
            callbacks=callbacks,
            max_epochs=max_epochs_dict[train_mode],
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            enable_progress_bar=True,
            enable_checkpointing=True,
        )

    def fit_warm_vae(self):

        train_mode = "warm_vae"

        # initialize model
        self.model = PartProtoVAE(**self.model_params)
        self.model.set_mode(train_mode)

        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.vae.parameters():
            param.requires_grad = True
        for param in self.model.prototype_block.proto_batch_norm.parameters():
            param.requires_grad = False
        self.model.prototype_block.prototype_vectors.requires_grad = False
        for param in self.model.last_layer.parameters():
            param.requires_grad = False

        # initialize checkpoint callback
        curr_ckpt_path = os.path.join(self.root_ckpt_dir, train_mode)

        # custom filename for callback because accuracy is not logged in this step
        filename = "{epoch}-{train_loss:.3f}"
        checkpoint_callback = getmodel_ckpt_callback(curr_ckpt_path,
                                                     filename=filename)

        custom_callbacks = [checkpoint_callback]

        tb_logger = TensorBoardLogger(self.root_logging_dir,
                                      name=train_mode,
                                      log_graph=False)

        self.initialize_trainer(train_mode, custom_callbacks, tb_logger)

        if use_validation:
            self.trainer.fit(self.model,
                             train_dataloaders=self.train_dataloader,
                             val_dataloaders=self.val_dataloader)
        else:
            self.trainer.fit(self.model,
                             train_dataloaders=self.train_dataloader)

        self.save_entire_model("warm_vae.pth")


    def fit_warm_proto(self):

        train_mode = "warm_proto"

        self.load_entire_model("warm_vae.pth")
        self.model.recon_coeff = recon_coeff

        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.vae.parameters():
            param.requires_grad = True
        for param in self.model.prototype_block.proto_batch_norm.parameters():
            param.requires_grad = True
        self.model.prototype_block.prototype_vectors.requires_grad = True
        for param in self.model.last_layer.parameters():
            param.requires_grad = False

        # initialize checkpoint callback
        curr_ckpt_path = os.path.join(self.root_ckpt_dir, train_mode)

        checkpoint_callback = getmodel_ckpt_callback(curr_ckpt_path)
        custom_callbacks = [checkpoint_callback]

        tb_logger = TensorBoardLogger(self.root_logging_dir,
                                      name=train_mode,
                                      log_graph=False)

        self.initialize_trainer(train_mode, custom_callbacks, tb_logger)

        if use_validation:
            self.trainer.fit(self.model,
                             train_dataloaders=self.train_dataloader,
                             val_dataloaders=self.val_dataloader)
        else:
            self.trainer.fit(self.model,
                             train_dataloaders=self.train_dataloader)
        self.save_entire_model("warm_proto.pth")

    def fit_joint(self, cycle_number=0):
        """
        Stage 3 of the training pipeline.

        For cycle 0 (single-cycle), loads warm_proto.pth.
        For cycle n>0 (multi-cycle), loads the
        last_layer output of the previous cycle and resumes Adam state from that
        cycle's joint optimizer checkpoint.
        """
        train_mode = "joint"

        # 1. Load the input model for this cycle.
        if cycle_number == 0:
            model_name = "warm_proto.pth"
        else:
            model_name = f"last_layer_cycle_{cycle_number - 1}.pth"
        self.load_entire_model(model_name)

        self.model.set_mode(train_mode, cycle_number)

        # 2. Freeze schedule. Last layer trained ONLY in Stage 5 (convex opt).
        for param in self.model.features.parameters():
            param.requires_grad = True
        for param in self.model.vae.parameters():
            param.requires_grad = True
        for param in self.model.prototype_block.proto_batch_norm.parameters():
            param.requires_grad = True
        self.model.prototype_block.prototype_vectors.requires_grad = True
        for param in self.model.last_layer.parameters():
            param.requires_grad = False

        # 1. initialize checkpoint callback
        curr_ckpt_path = os.path.join(self.root_ckpt_dir, f"cycle_{cycle_number}", train_mode)

        checkpoint_callback = getmodel_ckpt_callback(curr_ckpt_path)
        custom_callbacks = [checkpoint_callback]

        # 3. Trainer.
        logging_folder = os.path.join(self.root_logging_dir, f"cycle_{cycle_number}")
        tb_logger = TensorBoardLogger(logging_folder, name=train_mode, log_graph=False)
        self.initialize_trainer(train_mode, custom_callbacks, tb_logger)

        if use_validation:
            self.trainer.fit(self.model,
                             train_dataloaders=self.train_dataloader,
                             val_dataloaders=self.val_dataloader)
        else:
            self.trainer.fit(self.model, train_dataloaders=self.train_dataloader)

        self.save_entire_model(f"joint_cycle_{cycle_number}.pth")



    def push_cycle(self, cycle_number):
        """
        Stage 4: prototype projection. Replaces each learned prototype with
        the nearest training-image patch of its class, computed from the
        VAE posterior MEAN.
        """
        self.load_entire_model(f"joint_cycle_{cycle_number}.pth")

        push_prototypes(self.train_push_dataloader, self.model, cycle_number, self.params['save_prototypes'],
                            self.prototype_saving_dir,
                            self.model.prototype_shape)

        # 7. save model post-push
        self.save_entire_model(f"push_cycle_{cycle_number}.pth")


    def fit_last_layer(self, cycle_number):
        """
        Stage 5: convex optimization of the bias-free last layer with L1 sparsity.
        Everything except the last layer is frozen; loss
        reduces to CE + L1. Loads push_cycle_{cycle_number}.pth and produces
        last_layer_cycle_{cycle_number}.pth.
        """
        train_mode = "last_layer"

        # 1. load model from last joint cycle
        self.load_entire_model(f"push_cycle_{cycle_number}.pth")
        self.model.set_mode(train_mode, cycle_number)

        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.vae.parameters():
            param.requires_grad = False
        for param in self.model.prototype_block.proto_batch_norm.parameters():
            param.requires_grad = False
        self.model.prototype_block.prototype_vectors.requires_grad = False
        for param in self.model.last_layer.parameters():
            param.requires_grad = True

        # 2. initialize checkpoint callback
        curr_ckpt_path = os.path.join(self.root_ckpt_dir, f"cycle_{cycle_number}",train_mode)

        checkpoint_callback = getmodel_ckpt_callback(curr_ckpt_path)
        custom_callbacks = [checkpoint_callback]

        logging_folder = os.path.join(self.root_logging_dir, f"cycle_{cycle_number}")
        tb_logger = TensorBoardLogger(logging_folder, name=train_mode, log_graph=False)
        self.initialize_trainer(train_mode, custom_callbacks, tb_logger)

        if use_validation:
            self.trainer.fit(self.model,
                             train_dataloaders=self.train_dataloader,
                             val_dataloaders=self.val_dataloader)
        else:
            self.trainer.fit(self.model,
                             train_dataloaders=self.train_dataloader)

        # Cycle-indexed optimizer state for next cycle's resume (multi-cycle only).
        self.save_entire_model(f"last_layer_cycle_{cycle_number}.pth")


    def test_model(self, test_folder, test_mode, use_samples=True, cycle_number=None):
        """
        Run the test set through a saved model. The two valid cycle states:
          - cycle_number is None: testing a warm stage (warm_vae or warm_proto).
            Loads {test_mode}.pth from final_models_dir.
          - cycle_number is set: testing a cycle stage (joint, push, last_layer).
            Loads {test_mode}_cycle_{n}.pth.

        `use_samples` selects the inference mode:
          True  -> 40 posterior samples, majority vote (thesis Sec 4.3.7).
          False -> posterior mean (thesis Fig 5.2 "mean" baseline).
        """
        if cycle_number is None:
            model_name = f"{test_mode}.pth"
            logger_dir = self.root_logging_dir
            logger_name = test_mode
        else:
            model_name = f"{test_mode}_cycle_{cycle_number}.pth"
            logger_dir = os.path.join(self.root_logging_dir, f"cycle_{cycle_number}")
            logger_name = test_mode

        self.load_entire_model(model_name)
        self.model.use_samples = use_samples   # use samples from the bottleneck, not the mean
        self.model.test_folder = test_folder

        tb_logger = TensorBoardLogger(logger_dir, name=logger_name, log_graph=False)

        trainer = pl.Trainer(
            accelerator=self.params['accelerator'],
            devices=1,
            logger=tb_logger,
        )
        trainer.test(self.model, self.test_dataloader)

    def get_model(self):
        return self.model
