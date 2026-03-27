import subprocess
import logging
from datetime import datetime 
from argparse import ArgumentParser
from syne_tune.constants import ST_CHECKPOINT_DIR

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('--embed_lr', type=float)
    parser.add_argument('--head_lr', type=float)
    parser.add_argument('--tied_embed_lr', type=float)
    parser.add_argument('--tied_embed_init_std', type=float)
    parser.add_argument('--matrix_lr', type=float)
    parser.add_argument('--scalar_lr', type=float)
    parser.add_argument('--muon_momentum', type=float)
    parser.add_argument('--muon_backend_steps', type=int)
    parser.add_argument('--muon_momentum_warmup_start', type=float)
    parser.add_argument('--muon_momentum_warmup_steps', type=int)
    parser.add_argument('--beta1', type=float)
    parser.add_argument('--beta2', type=float)
    parser.add_argument('--adam_eps', type=float)
    parser.add_argument('--grad_clip_norm', type=float)
    parser.add_argument('--warmdown_iters', type=int)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument(f"--{ST_CHECKPOINT_DIR}", type=str)
    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    dict_args.pop(ST_CHECKPOINT_DIR)

 #   run_id_str = "_" + "_".join([f"{k}_{getattr(args, k)}" for k in dict_args.keys()])
    run_id_str = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    command = f"""RUN_ID=baseline{run_id_str} \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    EMBED_LR={args.embed_lr} \
    HEAD_LR={args.head_lr} \
    TIED_EMBED_LR={args.tied_embed_lr} \
    TIED_EMBED_INIT_STD={args.tied_embed_init_std} \
    MATRIX_LR={args.matrix_lr} \
    SCALAR_LR={args.scalar_lr} \
    MUON_MOMENTUM={args.muon_momentum} \
    MUON_BACKEND_STEPS={args.muon_backend_steps} \
    MUON_MOMENTUM_WARMUP_START={args.muon_momentum_warmup_start} \
    MUON_MOMENTUM_WARMUP_STEPS={args.muon_momentum_warmup_steps} \
    BETA1={args.beta1} \
    BETA2={args.beta2} \
    ADAM_EPS={args.adam_eps} \
    GRAD_CLIP_NORM={args.grad_clip_norm} \
    WARMDOWN_ITERS={args.warmdown_iters} \
    WARMUP_STEPS={args.warmup_steps} \
    VOCAB_SIZE=1024 \
    torchrun --standalone --nproc_per_node=4 train_gpt.py"""
    print(f"Launching command:\n{command}")
    process = subprocess.run(command, shell=True, check=True)
    print(f"Command finished with exit code {process.returncode}")

