# launch_height_simple.py
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, uniform, loguniform
from syne_tune.optimizer.baselines import  CQR

# hyperparameter search space to consider
config_space = {
  'embed_lr': loguniform(1e-2, 1.0),
  'head_lr': loguniform(1e-4, 1e-1),
  'tied_embed_lr': loguniform(1e-3, 1e-1),
  'tied_embed_init_std': loguniform(1e-4, 1e-2),
  'matrix_lr': loguniform(1e-3, 1e-1),
  'scalar_lr': loguniform(1e-3, 1e-1),
  'muon_momentum': uniform(0.9, 0.99),
  'muon_backend_steps': randint(1, 10),
  'muon_momentum_warmup_start': uniform(0.7, 0.95),
  'muon_momentum_warmup_steps': randint(100, 1000),
  'beta1': uniform(0.8, 0.99),
  'beta2': uniform(0.9, 0.999),
  'adam_eps': loguniform(1e-9, 1e-7),
  'grad_clip_norm': uniform(0.0, 1.0),
  'warmdown_iters': randint(500, 1500),
  'warmup_steps': randint(5, 50),
}
default_config = {
  'embed_lr': 0.6,
  'head_lr': 0.008,
  'tied_embed_lr': 0.05,
  'tied_embed_init_std': 0.005,
  'matrix_lr': 0.04,
  'scalar_lr': 0.04,
  'muon_momentum': 0.95,
  'muon_backend_steps': 5,
  'muon_momentum_warmup_start': 0.85,
  'muon_momentum_warmup_steps': 500,
  'beta1': 0.9,
  'beta2': 0.95,
  'adam_eps': 1e-8,
  'grad_clip_norm': 0.0,
  'warmdown_iters': 1200,
  'warmup_steps': 20,
}

tuner = Tuner(
    trial_backend=LocalBackend(entry_point='train_gpt.py',
                               binary='"torchrun --standalone --nproc_per_node=4',
                               num_gpus_per_trial=4),
    scheduler=CQR(
        config_space,
        metric='val_loss',
        points_to_evaluate=[default_config]
    ),
    stop_criterion=StoppingCriterion(max_wallclock_time=3600 * 24), # total runtime in seconds
    n_workers=1,  # how many trials are evaluated in parallel
)
tuner.run()
