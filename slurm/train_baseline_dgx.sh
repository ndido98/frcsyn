#!/bin/bash

#SBATCH --account=IscrC_LM-MAD_0
#SBATCH --partition=dgx_usr_preempt
#SBATCH --job-name=frcsyn-train_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --error=slurm_train_baseline_%A_%a.err
#SBATCH --output=slurm_train_baseline_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolo.didomenico@unibo.it

set -e

scratch_home=$CINECA_SCRATCH/frcsyn
project_dir=$scratch_home/frcsyn
datasets_dir=$scratch_home/datasets
output_dir=$scratch_home/job_$SLURM_JOB_ID

cd $scratch_home
echo "=== Pulling singularity image ==="
if [ ! -f pytorch_23.09-py3.sif ]; then
    singularity pull docker://nvcr.io/nvidia/pytorch:23.09-py3
fi
singularity exec --bind $project_dir:/mnt/frcsyn pytorch_23.09-py3.sif pip install -r /mnt/frcsyn/requirements.txt

echo "=== Copying datasets to $TMPDIR ==="
mkdir -p $TMPDIR/datasets
cp -r $datasets_dir/* $TMPDIR/datasets/

echo "=== Running training ==="
mkdir -p $TMPDIR/tmpdir
singularity exec \
    --bind $TMPDIR/datasets:/mnt/datasets,$project_dir:/mnt/frcsyn,$TMPDIR/tmpdir:/mnt/tmpdir \
    --nv pytorch_23.09-py3.sif \
    python /mnt/frcsyn/main.py fit \
    -c /mnt/frcsyn/experiments/train.yml -c /mnt/frcsyn/experiments/real.yml \
    --data.datasets.root /mnt/datasets --model.temp_dir /mnt/tempdir

echo "=== Done, copying back data ==="
mkdir -p $output_dir
cp -r $TMPDIR/tmpdir/* $output_dir/
