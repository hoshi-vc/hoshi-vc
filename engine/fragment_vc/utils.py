import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
  lr_lambda = cosine_schedule_with_warmup_lambda(num_warmup_steps, num_training_steps, num_cycles)
  return LambdaLR(optimizer, lr_lambda, last_epoch)

def cosine_schedule_with_warmup_lambda(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
):
  def lr_lambda(current_step: int):
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

  return lr_lambda
