#!/bin/bash
#SBATCH --job-name=my_little_job  # Job name
#SBATCH --time=20:00:00           # Time limit hrs:min:sec
#SBATCH -w gorgona5
#SBATCH -N 1                        # Number of nodes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x # all comands are also outputted

cd /scratch/larissa.gomide
module list
module avail
module load python3.12.1
module load cuda/11.8.0

source venv/bin/activate

export HOME="/minha_home"
export TRANSFORMERS_CACHE="/scratch/larissa.gomide/minha_home/.cache/huggingface"
export CLIP_CACHE="/scratch/larissa.gomide/minha_home/.cache/clip"
export HF_HOME="/scratch/larissa.gomide/minha_home/.cache/huggingface"
export XDG_CACHE_HOME="/scratch/larissa.gomide/minha_home/.cache"
export MPLCONFIGDIR='/scratch/larissa.gomide/minha_home/.matplotlib'

cd /home_cerberus/disk3/larissa.gomide/PKDD

python3 /home_cerberus/disk3/larissa.gomide/PKDD/polarized_description.py "/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_dataset.csv"|| echo "Erro ao executar polarized_description.py."

python3 /home_cerberus/disk3/larissa.gomide/PKDD/manda_email.py || echo "Erro ao executar manda_email.py."

hostname   # just show the allocated node

echo "Meu job terminou!" 