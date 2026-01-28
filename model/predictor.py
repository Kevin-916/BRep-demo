from torch.nn import Module, Parameter
import torch
from .generator import TopoGenerator
from .brep import brain_rep
from torch.nn import functional as F

class PredictionModel(Module):
    def __init__(self,args):
        super(PredictionModel, self).__init__()
        # Processing function
        self.GNN = brain_rep(args=args,
                         node_sz=args.node_sz,
                         time_series_sz=args.timeseries_sz,
                         corr_pearson_sz=args.node_feature_sz,
                         nlayers=args.layers,
                         dropout=args.dropout,
                         cluster_num=args.cluster_num,
                         pooling=args.pooling)
        self.generator = TopoGenerator(args=args,num_nodes=args.node_sz)

    def forward(self,timeseries, pearson):

        if self.training:
            node_feature = pearson
            corr, dae_h, dae_noise = self.generator(timeseries, pearson)
            y_pred = self.GNN(corr, node_feature)

        else:
            node_feature = pearson
            corr = self.generator(timeseries, pearson)
            y_pred = self.GNN(corr, node_feature)
            dae_h = None
            dae_noise = None

        return y_pred, dae_h, dae_noise
