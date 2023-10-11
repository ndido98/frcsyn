#!/bin/bash

#SBATCH --account=IscrC_LM-MAD
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=frcsyn-align_faces
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --error=slurm_align_faces_%A_%a.err
#SBATCH --output=slurm_align_faces_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolo.didomenico@unibo.it
#SBATCH --array=0-7

set -e

datasets=("Real/AgeDB" "Real/BUPT-BalancedFace" "Real/CASIA-WebFace" "Real/CFP-FP" "Real/FFHQ" "Real/ROF" "Synth/DCFace" "Synth/GANDiffFace")
image_size=112
margin=12
input="$WORK/frcsyn-datasets/${datasets[$SLURM_ARRAY_TASK_ID]}"
output="$WORK/frcsyn-datasets-${image_size}-m${margin}-aligned/${datasets[$SLURM_ARRAY_TASK_ID]}"

mkdir -p $output

module load profile/deeplrn
module load cuda/11.8
module load cudnn/8.4.0.27-11.6--gcc--11.3.0
module load python/3.10.8--gcc--11.3.0

cd $WORK/frcsyn
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3 ./align_faces.py --input "$input" --output "$output" -r -m $margin -s $image_size -a -n 32