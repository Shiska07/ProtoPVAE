
import torch
import torchsummary
from torch import nn
from torch.nn import functional as F

class EncoderBlock(nn.Module):
    def __init__(self, input_channels, latent_channels, dropout_rate=0.35):
        super(EncoderBlock, self).__init__()

        n_units = list()
        for i in range(2):
            val = int(input_channels/(2 ** (i + 1)))
            n_units.append(val)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv1 = nn.Conv2d(input_channels, n_units[0], kernel_size=1, stride=1,
                               padding="valid")
        self.bn1 = nn.BatchNorm2d(n_units[0])
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(n_units[0], n_units[1], kernel_size=1, stride=1,
                               padding="valid")
        self.bn2 = nn.BatchNorm2d(n_units[1])
        self.dropout3 = nn.Dropout2d(dropout_rate)
        self.conv3 = nn.Conv2d(n_units[1], latent_channels*2, kernel_size=1,
                               stride=1, padding="valid")


    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout3(x)
        x = self.conv3(x)
        return x

    def conv_info(self):
        return self.conv_feat


class DecoderBlock(nn.Module):
    def __init__(self, latent_channels, output_channels,
                 dropout_rate=0.35):
        super(DecoderBlock, self).__init__()

        n_units = []
        for i in reversed(range(2)):
            val = int(output_channels / (2 ** (i + 1)))
            n_units.append(val)

        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv_transpose1 = nn.ConvTranspose2d(latent_channels,
                                                  n_units[0],
                                                  kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(n_units[0])
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.conv_transpose2 = nn.ConvTranspose2d(n_units[0], n_units[1],
                                                  kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(n_units[1])
        self.dropout3 = nn.Dropout2d(dropout_rate)
        self.conv_transpose3 = nn.ConvTranspose2d(n_units[1], output_channels,
                                                  kernel_size=1, stride=1)

    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.bn1(self.conv_transpose1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn2(self.conv_transpose2(x)))
        x = self.dropout3(x)
        x = F.relu(self.conv_transpose3(x))
        return x


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VAE(nn.Module):

    def __init__(self,
                 prior_mu,
                 prior_std,
                 input_height,
                 latent_dim,
                 latent_channels,
                 features_out_dim,
                 dropout_rate=0.35,
                 init_weights=True):

        super(VAE, self).__init__()

        self.input_height = input_height
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels

        self.prior_mu = prior_mu
        self.prior_std = prior_std


        # VAE COMPONENTS
        self.encoder = EncoderBlock(input_channels=features_out_dim[1],
                                    latent_channels=latent_channels,
                                    dropout_rate=dropout_rate)

        self.decoder = DecoderBlock(latent_channels=latent_channels,
                                    output_channels=features_out_dim[1],
                                    dropout_rate=dropout_rate)

        if init_weights:
            self._initialize_weights()

    def sample(self, mu, log_var, n_samp):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu) + self.prior_mu,
                                       torch.ones_like(
                                           std) * self.prior_std)
        q = torch.distributions.Normal(mu, std)

        if n_samp > 1:
            batch_size = mu.shape[0]

            # sample 30 times for each datapoint
            mu_reshaped = mu.view(batch_size, -1, *self.latent_dim)
            mu_rep = mu_reshaped.repeat(1, n_samp, 1, 1, 1)
            mu_rep = mu_rep.view(-1, *self.latent_dim)

            std_reshaped = std.view(batch_size, -1, *self.latent_dim)
            std_rep = std_reshaped.repeat(1, n_samp, 1, 1, 1)
            std_rep = std_rep.view(-1, *self.latent_dim)

            mu_rep = mu_rep.to(device)
            std_rep = std_rep.to(device)

            # noise samples from a standard normal dist
            sampling_dist = torch.distributions.Normal(mu_rep, std_rep)
            z_n = sampling_dist.rsample()

        else:
            z_n = q.rsample()
        return p, q, z_n

    # step uses a sample size of 1
    def step(self, x, batch_idx=None):
        encoder_out = self.encoder(x)
        mu = encoder_out[:, :self.latent_channels, :, :]
        logvar = encoder_out[:, self.latent_channels:, :, :]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        p, q, z = self.sample(mu, logvar, n_samp=1)

        x_hat = self.decoder(z)

        return p, q, z, mu, x_hat

    # forward uses a sample size of > 1 and doesn't do reconstruction
    def forward(self, x, n_samp, batch_idx=None):
        encoder_out = self.encoder(x)
        mu = encoder_out[:, :self.latent_channels, :, :]
        logvar = encoder_out[:, self.latent_channels:, :, :]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        p, q, z_n = self.sample(mu, logvar, n_samp)

        return p, q, z_n, mu

    # initialize weights
    def _initialize_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':

    encoder_in_channels = 1024
    encoder = EncoderBlock(encoder_in_channels, 128)
    # print(encoder)
    torchsummary.summary(encoder, (encoder_in_channels, 7, 7))

    decoder = DecoderBlock(64, encoder_in_channels)
    # print(decoder)
    torchsummary.summary(decoder, (64, 7, 7))

