# launch_height_simple.py
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, uniform, loguniform
from syne_tune.optimizer.baselines import  CQR

# hyperparameter search space to consider
config_space = {
  'scalar_lr': loguniform(1e-5, 1e-1),
}

tuner = Tuner(
    trial_backend=LocalBackend(entry_point='launch_baseline.py'),
    scheduler=CQR(
        config_space,
        metric='val_loss',
    ),
    stop_criterion=StoppingCriterion(max_wallclock_time=3600), # total runtime in seconds
    n_workers=1,  # how many trials are evaluated in parallel
)
tuner.run()