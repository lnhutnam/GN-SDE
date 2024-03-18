from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# from bayesian_torch.layers.base_variational_layer import BaseVariationalLayer_
from torch.quantization.observer import MinMaxObserver
from torch.quantization.qconfig import QConfig

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
)
from torch_geometric.utils import spmm

from thirdparty.bayesian_torch.layers.base_variational_layer import (
    BaseVariationalLayer_,
)
from utils.bgcn_utils import gcn_norm


class BayesianGCNConv(MessagePassing, BaseVariationalLayer_):

    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        prior_mean: float = 0.0,  # Torch-Baysian
        prior_variance: float = 1.0,  # Torch-Baysian
        posterior_mu_init: float = 0.0,  # Torch-Baysian
        posterior_rho_init: float = -3.0,  # Torch-Baysian
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(
                f"'{self.__class__.__name__}' does not support "
                f"adding self-loops to the graph when no "
                f"on-the-fly normalization is applied"
            )

        # GCN specific initializations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None

        #         self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        # Bayesian specific initializations
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = (posterior_mu_init,)  # mean of weight
        self.posterior_rho_init = (posterior_rho_init,)
        self.bias = bias

        # Initialize Bayesian weights and biases
        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels))
        self.rho_weight = Parameter(torch.Tensor(out_channels, in_channels))
        self.register_buffer(
            "eps_weight", torch.Tensor(out_channels, in_channels), persistent=False
        )
        self.register_buffer(
            "prior_weight_mu", torch.Tensor(out_channels, in_channels), persistent=False
        )
        self.register_buffer(
            "prior_weight_sigma",
            torch.Tensor(out_channels, in_channels),
            persistent=False,
        )
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.rho_bias = Parameter(torch.Tensor(out_channels))
            self.register_buffer(
                "eps_bias", torch.Tensor(out_channels), persistent=False
            )
            self.register_buffer(
                "prior_bias_mu", torch.Tensor(out_channels), persistent=False
            )
            self.register_buffer(
                "prior_bias_sigma", torch.Tensor(out_channels), persistent=False
            )
        else:
            self.register_buffer("prior_bias_mu", None, persistent=False)
            self.register_buffer("prior_bias_sigma", None, persistent=False)
            self.register_parameter("mu_bias", None)
            self.register_parameter("rho_bias", None)
            self.register_buffer("eps_bias", None, persistent=False)

        self.init_parameters()
        self.quant_prepare = False

    def reset_parameters(self):
        super().reset_parameters()

        # Bayesian Weights Initialization
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)
        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)

        # Bayesian Biases Initialization (if bias is True)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0], std=0.1)

        # Reset cached GCN values
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        return_kl: bool = True,
    ) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(
                f"'{self.__class__.__name__}' received a tuple "
                f"of node features as input while this layer "
                f"does not support bipartite message passing. "
                f"Please try other layers such as 'SAGEConv' or "
                f"'GraphConv' instead"
            )

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                        self.flow,
                        x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # ---------- New BNN ---------
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        eps_weight = self.eps_weight.data.normal_()
        tmp_result = sigma_weight * eps_weight
        weight = self.mu_weight + tmp_result

        if return_kl:
            kl_weight = self.kl_div(
                self.mu_weight,
                sigma_weight,
                self.prior_weight_mu,
                self.prior_weight_sigma,
            )

        bias = None
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            bias = self.mu_bias + (sigma_bias * self.eps_bias.data.normal_())
            if return_kl:
                kl_bias = self.kl_div(
                    self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma
                )

        out = F.linear(x, weight, bias)
        out = self.propagate(
            edge_index, x=out, edge_weight=edge_weight, size=None
        )  # New
        if self.quant_prepare:
            # quint8 quantstub
            input = self.quint_quant[0](input)  # input
            out = self.quint_quant[1](out)  # output

            # qint8 quantstub
            sigma_weight = self.qint_quant[0](sigma_weight)  # weight
            mu_weight = self.qint_quant[1](self.mu_weight)  # weight
            eps_weight = self.qint_quant[2](eps_weight)  # random variable
            tmp_result = self.qint_quant[3](tmp_result)  # multiply activation
            weight = self.qint_quant[4](weight)  # add activatation

        if return_kl:
            if self.mu_bias is not None:
                kl = kl_weight + kl_bias
            else:
                kl = kl_weight

            return out, kl

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    # From Torch Baysian
    def prepare(self):
        self.qint_quant = nn.ModuleList(
            [
                torch.quantization.QuantStub(
                    QConfig(
                        weight=MinMaxObserver.with_args(
                            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
                        ),
                        activation=MinMaxObserver.with_args(
                            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
                        ),
                    )
                )
                for _ in range(5)
            ]
        )
        self.quint_quant = nn.ModuleList(
            [
                torch.quantization.QuantStub(
                    QConfig(
                        weight=MinMaxObserver.with_args(dtype=torch.quint8),
                        activation=MinMaxObserver.with_args(dtype=torch.quint8),
                    )
                )
                for _ in range(2)
            ]
        )
        self.dequant = torch.quantization.DeQuantStub()
        self.quant_prepare = True

    def init_parameters(self):
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=0.1)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=0.1)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0], std=0.1)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight, sigma_weight, self.prior_weight_mu, self.prior_weight_sigma
        )
        if self.mu_bias is not None:
            sigma_bias = torch.log1p(torch.exp(self.rho_bias))
            kl += self.kl_div(
                self.mu_bias, sigma_bias, self.prior_bias_mu, self.prior_bias_sigma
            )
        return kl
