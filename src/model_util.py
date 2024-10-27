import enum
import torch
import warnings

class InferenceType(enum.Enum):
    """
    Inference mode switch for your implementation.
    `MAP` simply predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2


class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler that calculates a different learning rate each gradient descent step.
    The default implementation keeps the original learning rate constant, i.e., does nothing.
    You can implement a custom schedule inside calculate_lr,
    and add+store additional attributes in __init__.
    You should not change any other parts of this class.
    """

    # TODO(2): Add and store additional arguments if you decide to implement a custom scheduler
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
        min_lr: float = 1e-6,
        max_lr: float = 1e-4,
        use_cyclical_lr: bool = False
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.use_cyclical_lr = use_cyclical_lr
        self.min_lr = min_lr
        self.max_lr = max_lr

        super().__init__(optimizer, last_epoch=-1)

    def calculate_lr(self, current_epoch: float, previous_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        current_epoch is the fractional epoch of SWA fitting, starting at 0.
        That is, an integer value x indicates the start of epoch (x+1),
        and non-integer values x.y correspond to steps in between epochs (x+1) and (x+2).
        previous_lr is the previous learning rate.

        This method should return a single float: the new learning rate.
        """
        # TODO(2): Implement a custom schedule if desired
        if self.use_cyclical_lr:
            c = self.steps_per_epoch
            t = 1/c * (((self._step_count - 1) % c) + 1)
            return (1-t) * self.max_lr + t * self.min_lr
        else:
            return previous_lr

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )
        return [
            self.calculate_lr(self.last_epoch / self.steps_per_epoch, group["lr"])
            for group in self.optimizer.param_groups
        ]