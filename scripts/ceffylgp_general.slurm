#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --account=taylor_group
#SBATCH --partition=debug
#SBATCH --time=30:00
#SBATCH --mail-user=william.g.lamb@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test.err

#Array setup here
#SBATCH --array=5

module purge
module load GCC/10.2.0 Autotools/20200321 Anaconda3/2019.10
source activate /gpfs52/home/lambwg/.conda/envs/ceffyl/

which conda
which python

python ceffylgp_general.py --Nfreqs $SLURM_ARRAY_TASK_ID --ceffyldir /home/lambwg/ng15yr_astro_interp/ceffyl_data/30f_fs{hd}_ceffyl --outdir /data/taylor_group/william_lamb/ng15_astro_interp/prod_runs/uniform-07a_new_n500_r100_f40/fshd_dmx_test/${SLURM_ARRAY_TASK_ID}f/ --gp_pkl /home/lambwg/ng15yr_astro_interp/spec_libraries/uniform-07a_new_n500_r100_f40/trained_gp_uniform-07a_n500_r100_med20230518_005932.pkl --var_gp_pkl /home/lambwg/ng15yr_astro_interp/spec_libraries/uniform-07a_new_n500_r100_f40/trained_gp_uniform-07a_n500_r100_std20230518_004033.pkl --resume
