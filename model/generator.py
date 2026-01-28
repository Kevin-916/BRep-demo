from typing import Union
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.distributions import Bernoulli
from .utils import inverse_sigmoid
from .knn import MLP_kNN
from .GCN_DAE import GraphDAE

class TopoGenerator(Module):
    def __init__(self,
                 args,
                 num_nodes: int,
                 ):
        super().__init__()

        self.num_nodes = num_nodes
        self.knn_num = args.knn_num
        self.knn = MLP_kNN(args,nlayers=args.knn_layer, isize=args.timeseries_sz,
                               hsize=args.timeseries_sz,osize=args.timeseries_sz,k=self.knn_num)

        self.dae = GraphDAE(args,in_dim=args.timeseries_sz,dropout=args.dropout)


    def forward(self, timeseries, pearson):
        probs = self.knn(timeseries,pearson)

        if self.training:
            h, noise = self.dae(probs, timeseries)
            return probs,h,noise
        else:
            return probs
        