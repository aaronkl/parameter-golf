import subprocess
import logging

from argparse import ArgumentParser

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('--scalar_lr', type=float)
    args, _ = parser.parse_known_args()
    command = f"""RUN_ID=baseline_lr_{args.scalar_lr} \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    SCALAR_LR={args.scalar_lr} \
    VOCAB_SIZE=1024 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py"""
    print(f"Launching command:\n{command}")
    process = subprocess.run(command, shell=True, check=True)
    print(f"Command finished with exit code {process.returncode}")

