import os
from subprocess import Popen, PIPE


datasets = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]
# datasets = [151507]
for data in datasets:
    batchfile = open(f'run_nichest_down_{data}.sh','w')
    batchfile.write('#!/bin/bash\n'+
                    '#SBATCH -p batch\n'+
                    '#SBATCH -N 1\n'+
                    '#SBATCH -n 16\n'+
                    '#SBATCH --mem=50G\n'+
                    '#SBATCH -t 02:00:00\n'+
                    f'#SBATCH --job-name run_nichest_down_{data}\n'+
                    f'#SBATCH --output run_nichest_down_{data}_o.txt\n'+
                    f'#SBATCH --error run_nichest_down_{data}_e.txt\n'+
                    'set -e\n'+
                    'cd /users/zgao62/nicheST/model/\n'+
                    'source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh\n'+
                    'conda activate quest\n' + 
                    f'python3 main_run_total.py --clustered --savedir ../results/{data} --downstream clustering --adata_path ../../data/zgao62/dlpfc_h5ad/{data}.h5ad\n')
    batchfile.close()

    process = Popen(['sbatch', f'run_nichest_down_{data}.sh'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()


# python3 main_run_total.py --savedir ../results/151507 --downstream clustering --clustered  --adata_path ../../data/zgao62/dlpfc_h5ad/151507.h5ad