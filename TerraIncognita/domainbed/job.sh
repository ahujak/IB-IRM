#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --job-name=PACS_1
#SBATCH --output=PACS_1.out
#SBATCH --error=PACS_1_error.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --mem=81Gb


# Load Modules and environements
module load python/3.6
source $HOME/invariance/bin/activate
module load httpproxy


cd $HOME/GitRepos/domainbed_ib/

python -m domainbed.scripts.sweep delete_incomplete\
       --algorithm IRM\
       --dataset PACS\
       --data_dir $HOME/scratch/data/\
       --output_dir $HOME/scratch/IRM_experiment/PACS_rerun/results_jmtd/1/\
       --command_launcher slurm_launcher\
       --skip_confirmation\
       --n_trials 3 \
       --n_hparams 40
       --default_hparams \
       --unique_test_env 0

python3 -m domainbed.scripts.sweep launch\
       --algorithm IRM\
       --dataset PACS\
       --data_dir $HOME/scratch/data/\
       --output_dir $HOME/scratch/IRM_experiment/PACS_rerun/results_jmtd/1/\
       --command_launcher slurm_launcher\
       --skip_confirmation\
       --n_trials 3 \
       --n_hparams 40
       --default_hparams \
       --unique_test_env 0
