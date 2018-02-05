#!/bin/bash
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 13:00:00
#SBATCH -p medium
#SBATCH --mem=8000
#SBATCH -o logs/hostname_%j.out
#SBATCH -e logs/hostname_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=petar_todorov@hms.harvard.edu
#SBATCH --array=1-100
export OMP_NUM_THREADS=4
uptime
python predict_background_ordreg.py settings_ordreg.json 1
uptime