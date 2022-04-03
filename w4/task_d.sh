#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 4096 # 4GB solicitados.
#SBATCH -p mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o .%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e .%x_%u_%j.err # File to which STDERR will be written
python task_d_UMAP.py