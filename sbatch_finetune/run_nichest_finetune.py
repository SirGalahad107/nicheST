import os
from subprocess import Popen, PIPE

batch_prefix = 'run_nichest'

# datasets = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]
datasets = [151507]
for data in datasets:
    batchfile = open(f'run_nichest_{data}.sh','w')
    batchfile.write('#!/bin/bash\n'+
                    '#SBATCH -p batch\n'+
                    '#SBATCH -N 1\n'+
                    '#SBATCH -n 12\n'+
                    '#SBATCH --mem=100G\n'+
                    '#SBATCH -t 02:00:00\n'+
                    f'#SBATCH --job-name run_nichest_{data}\n'+
                    f'#SBATCH --output run_nichest_{data}_o.txt\n'+
                    f'#SBATCH --error run_nichest_{data}_e.txt\n'+
                    'set -e\n'+
                    'cd /users/zgao62/nicheST/model/\n'+
                    'source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh\n'+
                    'conda activate quest\n' + 
                    f'python3 main_run.py --savedir ../results/{data} --preprocess --dec_type linear --adata_path ../../data/zgao62/dlpfc_h5ad/{data}.h5ad \n')
    batchfile.close()

    process = Popen(['sbatch', f'run_nichest_{data}.sh'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()