#!/bin/bash
#SBATCH --job-name=model_test_fscre_reg_1
#SBATCH --mail-user=sefie08@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=40GB
#SBATCH --time=15:00:00
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

cd /home/${USER}/projects/

module add Python/3.9.5-GCCcore-10.3.0

source ~/path/to/new/virtual/environment/bin/activate
pip install -U scikit-learn
pip install --upgrade transformers
pip install --upgrade accelerate
pip install sentencepiece
pip install pytesseract transformers datasets rouge-score nltk tensorboard py7zr --upgrade
pip install ipywidgets
pip install peft
pip install bitsandbytes
pip install evaluate
pip install trl
pip install torch
pip install torchvision
pip install --upgrade torch torchvision torchaudio
pip install huggingface-cli
pip install git+https://github.com/huggingface/trl
pip install --upgrade huggingface_hub
pip install -r requirements.txt
pip install scikit-learn

huggingface-cli login --token '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'

python FS-CRE/model_test_tacred_mas_random.py
