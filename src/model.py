# VAE implementation from pytorch_lightning bolts is used in this source code:
# https://github.com/Lightning-Universe/lightning-bolts/tree/master/pl_bolts/models/autoencoders
import os
import json

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.epoch_configs import max_epochs_dict
from configs.vae_configs import prior_mu, prior_std, n_samples
from configs.train_settings import class_specific, use_l1_mask, use_validation
from configs.lr_configs import warm_vae_optimizer_lrs, joint_optimizer_lrs, last_layer_optimizer_lr, \
     warm_protoL_optimizer_lrs
from configs.loss_configs import ce_coeff, l1_coeff, \
    clst_coeff, sep_coeff
from configs.proto_configs import input_height, input_channels,  \
    num_prototypes, n_classes, \
    prototype_activation_function

from utils.helpers import create_dir, get_average_losses, get_logs, \
     save_dict_as_json, get_accuracy

from model_components.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from model_components.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from model_components.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
from model_components.vae_components import VAE
from model_components.proto_components import PrototypeBlock

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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


class PartProtoVAE(LightningModule):
    def __init__(
        self,
        base_architecture,
        dropout_rate,
        kl_coeff,
        recon_coeff,
        latent_channels,
        sigmoid_shift,
        model_arc_dir,
        prototype_model_saving_dir,
        logging_dir,
        root_results_dir,
        proto_layer_rf_info,
        final_models_dir,
        init_weights=True
    ):

        super(PartProtoVAE, self).__init__()

        # saving hparams to model state dict
        self.save_hyperparameters()

        self.kl_coeff = kl_coeff
        self.recon_coeff = recon_coeff
        self.img_size = input_height
        self.latent_channels = latent_channels
        self.num_prototypes = num_prototypes
        self.prototype_shape = (num_prototypes, latent_channels, 1, 1)
        self.latent_dim = (latent_channels, 7, 7)
        self.num_classes = n_classes
        self.epsilon = 1e-4

        # prototype_activation_function could be 'log', 'linear',
        self.prototype_activation_function = prototype_activation_function

        self.base_architecture = base_architecture
        self.logging_dir = logging_dir
        self.model_arc_dir = model_arc_dir
        self.root_results_dir = root_results_dir
        self.final_models_dir = final_models_dir
        self.prototype_model_saving_dir = prototype_model_saving_dir

        # lists to store loses from each step
        # losses sores as tuple (rec_loss, kl_loss, total_loss)
        self.training_step_losses = []
        self.validation_step_losses = []

        # self.test_step_losses = []
        self.test_step_logits = []
        self.test_step_targets = []

        self.train_mode = None
        self.test_folder = None
        self.joint_cycle_number = -1

        self.use_samples = True  # toggle whether to use sample of the mean during training or inference

        # PRETRAINED FEATURE EXTRACTOR
        self.features = base_architecture_to_features[
            base_architecture](pretrained=True)

        # get input dim for features
        features_input_dim = (1, input_channels, input_height, input_height)
        dummy_input = torch.randn(features_input_dim)
        features_out_dim = (self.features(dummy_input)).size()

        # VAE COMPONENTS
        self.vae = VAE(prior_mu,
                       prior_std,
                       input_height,
                       self.latent_dim,
                       latent_channels,
                       features_out_dim,
                       dropout_rate)

        self.prototype_block = PrototypeBlock(n_classes,
                                              self.prototype_shape,
                                              latent_channels,
                                              prototype_activation_function,
                                              sigmoid_shift)

        self.last_layer = nn.Linear(self.num_prototypes, n_classes,
                                    bias=False)  # do not use bias

        self.proto_layer_rf_info = proto_layer_rf_info

        # to store conv info
        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

        if init_weights:
            self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(
            self.prototype_block.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def conv_info(self):
        self.kernel_sizes, self.strides, self.paddings = \
            self.features.conv_info()
        return self.kernel_sizes, self.strides, self.paddings

    # optional debugging helper
    def view_lr_info(self, optimizers):
        if self.trainer.global_rank == 0:
            print(f'Model LR info for mode: {self.train_mode}.')
            for i, param_group in enumerate(optimizers.param_groups):
                print(f'Parameter Group {i}:')
                print(f'Learning Rate: {param_group["lr"]}')

    # optional debugging helper
    def view_params_grad_info(self):
        if self.trainer.global_rank == 0:
            print(f'\nModel PARAM info for mode: {self.train_mode}.')
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(f'Parameter: {name} - Gradients: ON')
                else:
                    print(f'Parameter: {name} - Gradients: OFF')


    def configure_optimizers(self):
        if self.trainer.global_rank == 0:
            print(f"Configuring optimizers for mode: {self.train_mode}.\n")

        if self.train_mode == "warm_vae":
            warm_vae_specs = \
                [{'params': self.vae.parameters(),
                  'lr': warm_vae_optimizer_lrs['vae'], 'weight_decay': 1e-3}]
            optimizer = torch.optim.Adam(warm_vae_specs)


        elif self.train_mode == "warm_proto":
            warm_proto_specs = [{'params': self.vae.parameters(),
                  'lr': warm_protoL_optimizer_lrs['vae'], 'weight_decay': 1e-3},
                                {'params':
                                     self.prototype_block.proto_batch_norm.parameters(),
                                 'lr': warm_protoL_optimizer_lrs['proto_bnorm'],
                                 'weight_decay':
                                     1e-3},
                                {'params': self.prototype_block.prototype_vectors,
                                'lr': warm_protoL_optimizer_lrs[
                                    'prototype_vectors']}
                                ]
            optimizer = torch.optim.Adam(warm_proto_specs)


        elif self.train_mode == "joint":
            joint_specs = \
                [{'params': self.features.parameters(),
                  'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
                 {'params': self.vae.parameters(),
                  'lr': joint_optimizer_lrs['vae'], 'weight_decay': 1e-3},
                 {'params': self.prototype_block.proto_batch_norm.parameters(),
                  'lr': warm_protoL_optimizer_lrs['proto_bnorm'],
                  'weight_decay':
                      1e-3},
                 {'params': self.prototype_block.prototype_vectors,
                  'lr': joint_optimizer_lrs['prototype_vectors']}
                ]


            optimizer = torch.optim.Adam(joint_specs)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

            return {"optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "train_loss",
                        "frequency": 1
                        }
                    }

        elif self.train_mode == "last_layer":
            last_layer_specs = [{'params': self.last_layer.parameters(),
                                 'lr': last_layer_optimizer_lr}]
            optimizer = torch.optim.Adam(last_layer_specs)

        return optimizer


    def set_mode(self, mode=None, cycle_number=None):
        """
        Sets self.train_mode and self.joint_cycle_number atomically.
        For warm_vae/warm_proto, cycle_number is ignored and sentinel values
        (-2, -1) are used. For joint/last_layer, cycle_number must be provided.
        """
        self.train_mode = mode
        if mode == "warm_vae":
            self.joint_cycle_number = -2
        elif mode == "warm_proto":
            self.joint_cycle_number = -1
        elif mode in ("joint", "last_layer"):
            if cycle_number is None:
                raise ValueError(f"cycle_number required for mode={mode}")
            self.joint_cycle_number = cycle_number


    def on_fit_start(self):

        if self.trainer.global_rank == 0:
            print(f"\t\t***************** Mode = "
                  f"{self.train_mode}: CYCLE {self.joint_cycle_number}******************")

            # save model hyperparameters
            params_dict = dict()
            for key, val in self.hparams.items():
                params_dict[key] = val

            save_dict_as_json(params_dict, self.prototype_model_saving_dir)


    def get_clst_loss(self, min_distances, y):

        if class_specific:
            max_dist = (self.prototype_shape[1] * self.prototype_shape[2]*
                        self.prototype_shape[3])
            prototypes_of_correct_class = torch.t(
                self.prototype_block.prototype_class_identity[:, y])
            similarity_vals = max_dist - min_distances
            similarity_vals_with_corr_prototypes = similarity_vals * prototypes_of_correct_class
            inverted_distances, _ = torch.max(similarity_vals_with_corr_prototypes,dim=1)
            distances_to_closest_correct_prototype = max_dist - inverted_distances
            cluster_cost = torch.mean(distances_to_closest_correct_prototype)

        else:
            min_distance, _ = torch.min(min_distances, dim=1)
            cluster_cost = torch.mean(min_distance)
        return cluster_cost


    def get_sep_loss(self, min_distances, y):

        if class_specific:
            max_dist = (self.prototype_shape[1] * self.prototype_shape[2] *
                        self.prototype_shape[3])
            prototypes_of_correct_class = torch.t(
                self.prototype_block.prototype_class_identity[:, y])
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            similarity_vals = max_dist - min_distances
            similarity_vals_with_incor_prototypes = similarity_vals * prototypes_of_wrong_class


            inverted_distances, _ = torch.max(
                similarity_vals_with_incor_prototypes, dim=1)
            distances_to_closest_incor_prototype = max_dist - inverted_distances
            separation_cost = torch.mean(distances_to_closest_incor_prototype)

        else:
            return 0
        return separation_cost


    def get_l1_loss(self):
        if use_l1_mask:
            l1_mask = 1 - torch.t(self.prototype_block.prototype_class_identity)
            l1 = (self.last_layer.weight * l1_mask).norm(p=1)
        else:
            # sum of the absolute values of the elements
            l1 = self.last_layer.weight.norm(p=1)
        return l1


    # returns the most frequent prediction across samples
    def get_multi_sample_predictions(self, all_samp_preds):
        all_samp_preds = torch.stack(all_samp_preds, dim=1)
        final_pred, _ = torch.mode(all_samp_preds, dim=1)
        return final_pred
    

    def push_forward(self, x, use_samp=False):
        feat_out = self.features(x)

        if use_samp:
            # only a sample size of 1 can be used with push forward
            p, q, z, mu = self.vae(feat_out, n_samp=1)
            proto_layer_input, distances = self.prototype_block.push_forward(z)
        else:
            p, q, z_n, mu = self.vae(feat_out, n_samp=n_samples)
            # z_mu = self.get_sample_mean(z_n)
            proto_layer_input, distances = self.prototype_block.push_forward(mu)
        return proto_layer_input, distances


    # for testing and explanations only
    def forward(self, x, batch_idx=None, use_samp=True):
        feat_out = self.features(x)
        p, q, z_n, mu = self.vae(feat_out, n_samples, batch_idx)

        # prototype block forward
        if use_samp:
            prototype_activations, distances, min_distances = \
                    self.prototype_block(z_n)
        else:
            prototype_activations, distances, min_distances = \
                self.prototype_block(mu)

        logits = self.last_layer(prototype_activations)
        return logits, distances, min_distances


    def step(self, batch, batch_idx=None):

        x, y = batch
        feat_out = self.features(x)
        p, q, z_n, mu, decoder_out = self.vae.step(feat_out)

        # get prototype layer input and distances
        prototype_activations, min_distances = self.prototype_block.step(z_n)
        logits = self.last_layer(prototype_activations)

        recon_loss = F.mse_loss(decoder_out, feat_out)
        kl = (torch.distributions.kl_divergence(q, p)).mean()

        cross_entropy = F.cross_entropy(logits, y)
        acc = get_accuracy(logits, y)
        cluster_cost = self.get_clst_loss(min_distances, y)
        separation_cost = self.get_sep_loss(min_distances, y)
        l1_loss = self.get_l1_loss()

        # adjust coefficients according to the stage
        if self.train_mode == "warm_vae":
            coeffs = [self.recon_coeff, self.kl_coeff, 0, 0, 0, 0]

        elif self.train_mode == "warm_proto":
            coeffs = [self.recon_coeff, self.kl_coeff, ce_coeff, clst_coeff, sep_coeff,
                      l1_coeff]
        else:
            coeffs = [self.recon_coeff, self.kl_coeff, ce_coeff, clst_coeff, sep_coeff,
                      l1_coeff]

        # loss that is backpropagated for unfrozen components
        loss = (coeffs[0] * recon_loss
                    + coeffs[1] * kl
                    + coeffs[2] * cross_entropy
                    + coeffs[3] * cluster_cost
                    + coeffs[4] * separation_cost
                    + coeffs[5] * l1_loss)

        # loss for all components
        total_loss = (self.recon_coeff * recon_loss
                    + self.kl_coeff * kl
                    + ce_coeff * cross_entropy
                    + clst_coeff * cluster_cost
                    + sep_coeff * separation_cost
                    + l1_coeff * l1_loss)

        step_losses = [recon_loss.item(),
                       kl.item(),
                       cross_entropy.item(),
                       cluster_cost.item(),
                       separation_cost.item(),
                       l1_loss.item(),
                       loss.item(),
                       total_loss.item(),
                       acc]

        logs = get_logs(step_losses)
        return logits, loss, logs, decoder_out

    '''
    Here we return the loss and logs for a single batch.
    '''
    def training_step(self, batch, batch_idx):
        _, loss, logs, _ = self.step(batch, batch_idx)
        train_logs = dict()
        for key, val in logs.items():
            new_key = "train_" + str(key)
            train_logs[new_key] = val

        self.log("train_loss", train_logs["train_loss"], on_step=False, on_epoch=True,
                 sync_dist=True)

        # don't lpg validation accuracy in warm_vae mode
        if self.train_mode != "warm_vae":
            self.log("train_acc", train_logs["train_acc"], on_step=False, on_epoch=True,
                     sync_dist=True)
        self.training_step_losses.append(logs)
        return loss


    def validation_step(self, batch, batch_idx):
        if use_validation:
            _, loss, logs, _ = self.step(batch, batch_idx)

            val_logs = dict()
            for key, val in logs.items():
                new_key = "val_" + str(key)
                val_logs[new_key] = val

            self.log("val_loss", val_logs["val_loss"], on_step=False, on_epoch=True,
                sync_dist=True)

            # don't lpg validation accuracy in warm_vae mode
            if self.train_mode != "warm_vae":
                self.log("val_acc", val_logs["val_acc"], on_step=False, on_epoch=True,
                         sync_dist=True)
            self.validation_step_losses.append(logs)
            return loss


    def test_step(self, batch, batch_idx):
        x, y = batch

        logits, distances, min_distances = self.forward(x, batch_idx,
                                                        use_samp=self.use_samples)
        if self.use_samples:
            # forward returned (batch * n_samples, n_classes); replicate each
            # label n_samples times to align. Per-sample predictions are
            # reduced by majority vote in on_test_epoch_end (thesis Sec 4.3.7).
            y_rep = y.view(-1, 1).repeat(1, n_samples).view(-1)
            self.test_step_targets.append(y_rep)
        else:
            # mean path: forward returned (batch, n_classes)
            self.test_step_targets.append(y)

        self.test_step_logits.append(logits)


    def on_train_epoch_end(self):
        avg_metric_dict = get_average_losses(
            self.training_step_losses)

        if self.global_rank == 0:
            tag = "train"
            print(f"\nTRAINING Epoch[{self.current_epoch}] GLOBAL RANK[{self.trainer.global_rank}]:")
            for key, val in avg_metric_dict.items():
                print(f"{key} : {val:0.4f}")

                if self.train_mode in ["last_layer"]:
                    self.logger.experiment.add_scalars(key, {tag: val}, self.current_epoch + max_epochs_dict["joint"] + 1)
                else:
                    self.logger.experiment.add_scalars(key, {tag: val},
                                                       self.current_epoch)

            print("\n")
            train_dir = os.path.join(self.root_results_dir, "train_results",
                                   self.train_mode)
            create_dir(train_dir)
            if self.train_mode == "joint":
                file_path = os.path.join(train_dir,
                                              f"cycle_"
                                              f"{self.joint_cycle_number}_0_epoch_"
                                              f"{self.current_epoch}.json")
            elif self.train_mode == "last_layer":
                file_path = os.path.join(train_dir,
                                         f"cycle_"
                                         f"{self.joint_cycle_number}_1_epoch_"
                                         f"{self.current_epoch}.json")
            else:
                file_path = os.path.join(train_dir,
                                         f"cycle_"
                                         f"{self.joint_cycle_number}_-1_epoch_"
                                         f"{self.current_epoch}.json")

            with open(file_path, 'w') as json_file:
                json.dump(avg_metric_dict, json_file, indent=4)
        self.training_step_losses.clear()


    def on_validation_epoch_end(self):
        if use_validation:
            avg_metric_dict = get_average_losses(
                self.validation_step_losses)

            if self.global_rank == 0:
                tag = "val"
                print(f"\nVALIDATION Epoch[{self.current_epoch}] GLOBAL RANK[{self.trainer.global_rank}]:")
                for key, val in avg_metric_dict.items():
                    print(f"{key} : {val:0.4f}")

                    if self.train_mode in ["last_layer"]:
                        self.logger.experiment.add_scalars(key, {tag: val}, self.current_epoch + max_epochs_dict["joint"] + 1)
                    else:
                        self.logger.experiment.add_scalars(key, {tag: val},
                                                           self.current_epoch)
                print("\n")
                val_dir = os.path.join(self.root_results_dir, "val_results",
                                       self.train_mode)
                create_dir(val_dir)
                if self.train_mode == "joint":
                    file_path = os.path.join(val_dir,
                                             f"cycle_"
                                             f"{self.joint_cycle_number}_0_epoch_"
                                             f"{self.current_epoch}.json")
                elif self.train_mode == "last_layer":
                    file_path = os.path.join(val_dir,
                                             f"cycle_"
                                             f"{self.joint_cycle_number}_1_epoch_"
                                             f"{self.current_epoch}.json")
                else:
                    file_path = os.path.join(val_dir,
                                             f"cycle_"
                                             f"{self.joint_cycle_number}_-1_epoch_"
                                             f"{self.current_epoch}.json")
                with open(file_path, 'w') as json_file:
                    json.dump(avg_metric_dict, json_file, indent=4)

            self.validation_step_losses.clear()


    def on_test_epoch_end(self):

        test_dir = os.path.join(self.root_results_dir, f"test_results",
                                self.test_folder)
        create_dir(test_dir)

        epoch_logits = torch.cat(self.test_step_logits, dim=0)
        epoch_targets = torch.cat(self.test_step_targets, dim=0)

        # accuracy and confusion matrix for mean and combined samples
        if self.use_samples:

            epoch_logits = epoch_logits.view(-1, n_samples, n_classes)
            epoch_targets = epoch_targets.view(-1, n_samples)

            # store predicitons from all sample sets to pick the most frequent one
            all_sample_preds = []

            for i in range(n_samples):
                sample_logits = epoch_logits[:, i, :] # all logits for ith samples
                sample_targets = epoch_targets[:, i]

                sample_preds = torch.argmax(F.softmax(sample_logits, dim=1), dim=1)
                all_sample_preds.append(sample_preds)

                # get acuracy
                sample_acc = get_accuracy(sample_logits, sample_targets)
                print(f"Test acc of {i}th sample: {sample_acc:0.5}")

                # save
                test_fname = f"test_{self.train_mode}_c{self.joint_cycle_number}_samp_no{i}_acc_{sample_acc:0.3}.txt"
                file_path = os.path.join(test_dir, test_fname)
                with open(file_path, 'w') as file:
                    file.write(f'test_acc : {sample_acc:0.7f}')


            multi_samp_preds = self.get_multi_sample_predictions(all_sample_preds)
            multi_samp_acc = get_accuracy(multi_samp_preds, epoch_targets[:, 0])
            print(f"Test acc of most freq prediction from samples: {multi_samp_acc:0.5}")

            # save multi_samp_acc and conf matrix
            test_fname = f"test_{self.train_mode}_c" \
                            f"{self.joint_cycle_number}_multisamp_vote_acc" \
                            f"_{multi_samp_acc:0.3}.txt"
            file_path = os.path.join(test_dir, test_fname)
            with open(file_path, 'w') as file:
                file.write(f'test_acc : {multi_samp_acc:0.7f}')

        else:
            acc = get_accuracy(epoch_logits, epoch_targets)
            print(f'\nTest acc: {acc:0.5}\n')

            test_fname = f"test_{self.train_mode}_c{self.joint_cycle_number}_use_mean_acc_{acc:0.3}.txt"
            file_path = os.path.join(test_dir, test_fname)
            with open(file_path, 'w') as file:
                file.write(f'test_acc : {acc:0.7f}')

        self.test_step_logits.clear()
        self.test_step_targets.clear()

