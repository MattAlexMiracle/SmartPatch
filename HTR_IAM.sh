#!/bin/bash
#SBATCH --job-name=HTR_IAM_Generation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/%x-%j-on-%N.out
#SBATCH -e /home/%u/%x-%j-on-%N.err
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00

export WORKON_HOME==/cluster/`whoami`/.python_cache

echo "Your job is running on" $(hostname)
pip3 install --user python-Levenshtein
# copy to /scratch/qi69dube
mkdir /scratch/qi69dube
cp -nr /cluster/qi69dube/dataset.tar.gz /scratch/qi69dube/dataset.tar.gz
tar xkf /scratch/qi69dube/dataset.tar.gz -C /scratch/qi69dube/
# -1 stands for: continue with the last saved-model
# This training run uses the actual IAM dataset to train
cd Benchmark/research-seq2seq-HTR/
python3 main_torch_latest.py -1 HTR_REAL_weights IAM_REAL
