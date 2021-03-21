import torch
import numpy as np
import torch.nn as nn
import pdb

# https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L622
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement
            # Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class Decoder(nn.Module):
    """ Decoder conditioned by adding.

    Example configuration:
        z_dim: 128
        hidden_size: 256
        n_blocks: 5
        out_dim: 3  # we are outputting the gradient
        sigma_condition: True
        xyz_condition: True
    """

    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.z_dim
        self.dim = dim = cfg.dim + 1
        self.out_dim = out_dim = cfg.out_dim
        self.hidden_size = hidden_size = cfg.hidden_size
        self.n_blocks = n_blocks = cfg.n_blocks

        # Network modules
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(dim, hidden_size))
        for _ in range(n_blocks):
            self.blocks.append(nn.Linear(hidden_size, hidden_size))
        self.blocks.append(nn.Linear(hidden_size, out_dim))
        self.act = Sine()

        # Initialization
        self.apply(sine_init)
        self.blocks[0].apply(first_layer_sine_init)

    # This should have the same signature as the sig condition one
    def forward(self, x, c_dims):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz)
        :param c: (bs, self.zdim + 1) Shape latent code + sigma
        TODO: will ignore [c] for now
        :return: (bs, npoints, self.dim) Gradient (self.dim dimension)
        """
        batch_size, num_points, D = x.size()
        sigmas = c_dims[:, -1]  # (bs)
        sigmas = sigmas.view(-1, 1, 1)  # (bs, 1,1)
        sigmas = sigmas.expand(batch_size, num_points, 1)  # (bs, num pts, 1)
        x = torch.cat([x, sigmas], dim=2)  # (bs, num_pts, 4)
        net = x
        for block in self.blocks[:-1]:
            net = self.act(block(net))
        out = self.blocks[-1](net)
        return out

