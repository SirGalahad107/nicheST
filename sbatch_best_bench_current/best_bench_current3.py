import os
from subprocess import Popen, PIPE

batch_prefix = 'best_bench_current3'

datasets = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]
# hvgs = [5000]

best_drop = 0.180
best_knn = 8
best_seed = 1209
best_enc_dims = '512 256 128'
best_dec_dims = '128 256 512'
best_pos_weight = 0.464
best_neg_weight = 1.700
best_gnn = 'gat'
best_model_k = 1
best_hvg = 3000


for data in datasets:
    batchfile = open(f'{batch_prefix}_{data}.sh','w')
    batchfile.write('#!/bin/bash\n'+
                    '#SBATCH -p gpu\n'+
                    '#SBATCH -N 1\n'+
                    '#SBATCH -n 6\n'+
                    "#SBATCH --gres=gpu:1\n"+
                    '#SBATCH --gres-flags=enforce-binding\n'+
                    '#SBATCH --mem=95G\n'+
                    '#SBATCH -t 24:00:00\n'+
                    f'#SBATCH --job-name {batch_prefix}_{data}\n'+
                    f'#SBATCH --output {batch_prefix}_{data}_o.txt\n'+
                    f'#SBATCH --error {batch_prefix}_{data}_e.txt\n'+
                    'set -e\n'+
                    'cd /users/zgao62/nicheST/model/\n'+
                    'source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh\n'+
                    'conda activate r_env \n' + 
                    f'python3 main_run_best_bench.py --save_dir ../{batch_prefix}/{data} --dropout {best_drop} --knn {best_knn} --seed {best_seed} --enc_dims {best_enc_dims} --dec_dims {best_dec_dims} --contra_pos_weight {best_pos_weight} --contra_neg_weight {best_neg_weight} --gnn {best_gnn} --model_k {best_model_k} --hvg {best_hvg} --downstream clustering --adata_path /users/zgao62/data/zgao62/dlpfc_h5ad/data/{data}.h5ad \n')
    batchfile.close()

    process = Popen(['sbatch', f'{batch_prefix}_{data}.sh'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()