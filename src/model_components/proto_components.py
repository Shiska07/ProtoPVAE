import torch
from torch import nn
from torch.nn import functional as F

'''
Some components of the following implementation were obtained from: https://github.com/cfchen-duke/ProtoPNet
'''

def shifted_sigmoid(x, shift=0.0):
    return 1/(1+torch.exp(-x + shift))

class PrototypeBlock(nn.Module):
    def __init__(self,
                 n_classes,
                 prototype_shape,
                 latent_channels,
                 prototype_activation_function,
                 sigmoid_shift,
                 init_weights=True):
        super(PrototypeBlock, self).__init__()

        self.n_classes = n_classes
        self.prototype_shape = prototype_shape
        self.latent_channels = latent_channels
        self.num_prototypes = prototype_shape[0]
        self.prototype_activation_function = prototype_activation_function
        self.shift = sigmoid_shift
        self.epsilon = 1e-4

        assert (self.num_prototypes % n_classes == 0)

        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = nn.Parameter(torch.zeros(self.num_prototypes,
                                                                 n_classes),
                                                     requires_grad=False)

        self.num_prototypes_per_class = self.num_prototypes // n_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        # PROTOTYPE AND CLASSIFIER COMPONENTS
        self.proto_batch_norm = nn.BatchNorm2d(latent_channels)
        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(prototype_shape),
                                 requires_grad=False)

        if init_weights:
            self._initialize_weights()


    def _l2_convolution(self, x):

        # apply self.prototype_vectors as l2-convolution filters on input x
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances

    def min_distances(self, distances):
        min_distances = -F.max_pool2d(-distances, kernel_size=(
        distances.size()[2], distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        return min_distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        return self.prototype_activation_function(distances)

    def push_forward(self, x):
        x = self.proto_batch_norm(x)
        proto_layer_input = shifted_sigmoid(x, self.shift)
        distances = self._l2_convolution(proto_layer_input)
        return proto_layer_input, distances

    def forward(self, x):
        x = self.proto_batch_norm(x)
        proto_layer_input = shifted_sigmoid(x, self.shift)
        distances = self._l2_convolution(proto_layer_input)
        min_distances = self.min_distances(distances)
        prototype_activations = self.distance_2_similarity(min_distances)
        return prototype_activations, distances, min_distances

    def step(self, x):
        x = self.proto_batch_norm(x)
        proto_layer_input = shifted_sigmoid(x, self.shift)
        distances = self._l2_convolution(proto_layer_input)
        min_distances = self.min_distances(distances)
        prototype_activations = self.distance_2_similarity(min_distances)
        return prototype_activations, min_distances

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
