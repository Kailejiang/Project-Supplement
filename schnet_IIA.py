from typing import Callable, Dict, Union, Optional, List
import torch
from torch import nn
import schnetpack.properties as properties
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus
import schnetpack.nn as snn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GATConv
import numpy as np


__all__ = ["SchNet", "SchNetInteraction"]

def SMU(x: torch.Tensor, alpha: float = 0.01, mu: float = 2.5) -> torch.Tensor:
    """
    Applies the Softmax Unit (SMU) activation function.

    Args:
        x (torch.Tensor): Input tensor.
        alpha (float, optional): Hyperparameter, defaults to 0.01.
        mu (float, optional): Hyperparameter, defaults to 2.5.

    Returns:
        torch.Tensor: Output tensor after applying the SMU activation.
    """
    mu_tensor = torch.tensor(mu, requires_grad=True)
    return ((1 + alpha) * x + (1 - alpha) * x * torch.erf(mu_tensor * (1 - alpha) * x)) / 2

class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        #activation: Callable = shifted_softplus,
        activation: Callable = SMU,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        x = self.f2out(x)
        return x


class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems"""

    def __init__(
        self,
            n_atom_basis: int,
            n_interactions: int,
            radial_basis: nn.Module,
            cutoff_fn: Callable,
            n_filters: int = None,
            shared_interactions: bool = False,
            activation: Union[Callable, nn.Module] = shifted_softplus,
            nuclear_embedding: Optional[nn.Module] = None,
            electronic_embeddings: Optional[List] = None,
            n_heads: int = 1,  # 多头注意力的数量
            out_channels: int = None  # 输出特征维度
    ):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff

        # initialize embeddings
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(400, n_atom_basis)
        self.embedding = nuclear_embedding
        if electronic_embeddings is None:
            electronic_embeddings = []
        electronic_embeddings = nn.ModuleList(electronic_embeddings)

        self.electronic_embeddings = electronic_embeddings

        # initialize interaction blocks
        self.interactions = snn.replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )


    def forward(self, inputs: Dict[str, torch.Tensor]):

        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        # compute pair features
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # compute initial embeddings
        x = self.embedding(atomic_numbers)
        for embedding in self.electronic_embeddings:
            x = x + embedding(x, inputs)

        dimension = x.shape[1]

        # compute interaction blocks and update atomic embeddings
        for interaction in self.interactions:
            gconv = GATConv(dimension, dimension, heads=10, concat=False).to("cuda:0")
            edge_index = torch.stack([idx_i, idx_j], dim=0).to("cuda:0")  # 构建边索引

            a = gconv(x, edge_index).to("cuda:0")  # 通过图注意力机制计算a
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v + a

        # collect results
        inputs["scalar_representation"] = x

        return inputs

    def get_embedding_weights(self):
        return self.embedding.weight.data.clone()

    def load_embedding_weights(self, weights):
        self.embedding.weight.data.copy_(weights)
        print("embedding weights loaded")
