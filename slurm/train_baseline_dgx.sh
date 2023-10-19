#!/bin/bash

#SBATCH --account=IscrC_LM-MAD_0
#SBATCH --partition=dgx_usr_preempt
#SBATCH --job-name=frcsyn-train_baseline
#SBATCH --nodes=1
#SBATCH --nodelist=dgx02
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --error=slurm_train_baseline_%A_%a.err
#SBATCH --output=slurm_train_baseline_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nicolo.didomenico@unibo.it

set -e

scratch_home=$CINECA_SCRATCH/frcsyn
project_dir=$scratch_home/frcsyn
datasets_dir=/raid/scratch_local/frcsyn/datasets
output_dir=$scratch_home/job_$SLURM_JOB_ID

cd $scratch_home
echo "=== Opening enroot container ==="
export ENROOT_CACHE_PATH=/raid/scratch_local/frcsyn/tmp/enroot-cache/group-$(id -g)
export ENROOT_DATA_PATH=/raid/scratch_local/frcsyn/tmp/enroot-data/user-$(id -u)
export ENROOT_RUNTIME_PATH=/raid/scratch_local/frcsyn/tmp/enroot-runtime/user-$(id -u)
export ENROOT_MOUNT_HOME=y
export NVIDIA_DRIVER_CAPABILITIES=all

echo "=== Running training ==="
mkdir -p $TMPDIR/tmpdir
enroot start \
    --mount $datasets_dir:/workspace/datasets --mount $project_dir:/workspace/frcsyn --mount $TMPDIR/tmpdir:/workspace/tmpdir \
    --root \
    --env NVIDIA_DRIVER_CAPABILITIES \
    --rw \
    pytorch2309 \
    sh -c 'python /workspace/frcsyn/main.py fit -c /workspace/frcsyn/experiments/train.yml -c /workspace/frcsyn/experiments/real.yml --trainer.strategy ddp'

echo "=== Done, copying back data ==="
mkdir -p $output_dir
cp -r $TMPDIR/tmpdir/* $output_dir/
