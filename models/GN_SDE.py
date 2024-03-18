import torch
import torch.nn as nn
import torch.nn.functional as F

from core.gn_sde import LatentGraphSDE
from models.layers.basic_gcn import BasicGCN


def GNSDE_Block(
    edge_index,
    num_feats,
    n_dimension: int,
    n_classes: int,
    device: torch.device,
    act_func=[F.relu, nn.Softplus(), nn.Softplus(), nn.Softmax()],
    dropout=[0.4, 0.2, 0.2, 0.0],
):
    network_in = BasicGCN(
        edge_index=edge_index,
        in_feats=num_feats,
        out_feats=n_dimension,
        activation=act_func[0],
        dropout=dropout[0],
    ).to(device)

    drif_function = nn.Sequential(
        BasicGCN(
            edge_index=edge_index,
            in_feats=n_dimension,
            out_feats=n_dimension,
            activation=act_func[1],
            dropout=dropout[1],
        ),
        BasicGCN(
            edge_index=edge_index,
            in_feats=n_dimension,
            out_feats=n_dimension,
            activation=act_func[2],
            dropout=dropout[2],
        ),
    ).to(device)

    network_out = nn.Sequential(
        nn.Flatten(),
        BasicGCN(
            edge_index=edge_index,
            in_feats=n_dimension,
            out_feats=n_classes,
            activation=act_func[3],
            dropout=dropout[3],
        ).to(device),
    ).to(device)
