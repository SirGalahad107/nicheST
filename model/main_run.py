import argparse
import numpy as np
import os
import torch
import pickle 
from preprocess import *
import random
import time
from train import NicheST_trainer
import optuna

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main():
    parser = argparse.ArgumentParser(description='NicheST new model')

    ## preprocess and directory
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--adata_path', type=str, default=None)
    parser.add_argument('--preprocess', action='store_true', help='Preprocessing spatial data')
    parser.add_argument('--downstream', type=str, default=None)
    
    parser.add_argument('--knn', type=int, default=6)
    parser.add_argument('--pca', type=bool, default=False)
    parser.add_argument('--graph_build', type=str, default='grid')
    parser.add_argument('--model_k', type=int, default=1, help='Number of hops for the aggregated neighbors')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--hvg', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    
    ## loss weight related
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--contra_pos_weight', type=float, default=1.0)
    parser.add_argument('--contra_neg_weight', type=float, default=1.0)

    ## model related
    parser.add_argument('--att_heads', type=int, default=1) 
    parser.add_argument('--att_thresh', type=float, default=0.25) 
    parser.add_argument('--enc_dims', type=list, default=[512, 128]) 
    parser.add_argument('--dec_dims', type=list, default=[128, 512])  
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--recon_loss_type', type=str, default='mse')
    parser.add_argument('--norm', type=str, default='layernorm')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--contra_type', type=str, default='mine')
    parser.add_argument('--gnn', type=str, default='gcn')


    parser.add_argument('--epochs', type=int, default=600)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    set_seed(args.seed)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    print(f"The saving directory set to {args.savedir}", flush=True)


    if args.preprocess:
        adata = sc.read_h5ad(args.adata_path)
        # print(adata)
        adata = filter_adata(adata)
        adata = preprocess(args, adata)

    else:
        adata = sc.read_h5ad(os.path.join(args.savedir, 'adata_preprocessed.h5ad'))

    

    param = vars(args)
    
    feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list = prepare_graph_data([adata], param)

    nicheST_trainer = NicheST_trainer(feature_list[0].shape[1], param)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        nicheST_trainer.model.load_state_dict(checkpoint)

    nicheST_trainer.train(feature_list, edge_ind_list, sub_node_sample_list, sub_edge_ind_sample_list)



    if args.downstream == 'clustering':
        pass


if __name__ == '__main__':
    main()