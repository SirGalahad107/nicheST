import os
from subprocess import Popen, PIPE

batch_prefix = 'run_ftall'

# datasets = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]
# hvgs = [5000]
# for hvg in hvgs:
batchfile = open(f'{batch_prefix}.sh','w')
batchfile.write('#!/bin/bash\n'+
                '#SBATCH -p batch\n'+
                '#SBATCH -N 1\n'+
                '#SBATCH -n 64\n'+
                '#SBATCH --mem=492G\n'+
                '#SBATCH -t 24:00:00\n'+
                f'#SBATCH --job-name {batch_prefix}\n'+
                f'#SBATCH --output {batch_prefix}_o.txt\n'+
                f'#SBATCH --error {batch_prefix}_e.txt\n'+
                'set -e\n'+
                'cd /users/zgao62/nicheST/model/\n'+
                'source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh\n'+
                'conda activate quest\n' + 
                f'python3 main_run_ftall.py --epochs 200 --save_dir ../results_ftall/ --downstream clustering --adata_path ../../data/zgao62/dlpfc_h5ad/ \n')
batchfile.close()

process = Popen(['sbatch', f'{batch_prefix}.sh'], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()