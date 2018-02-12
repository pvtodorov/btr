#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH -p short
#SBATCH --mem=4000
#SBATCH -o logs/hostname_%j.out
#SBATCH -e logs/hostname_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=petar_todorov@hms.harvard.edu
#SBATCH --array=1-100
export OMP_NUM_THREADS=1
uptime
python predict.py run_settings/settings_LPOCV_BM36_Ordinal.json
uptime