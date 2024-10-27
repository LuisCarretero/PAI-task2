import abc
import collections
import enum
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import torch.optim
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt

from model_util import InferenceType, SWAGScheduler

EXTENDED_EVALUATION = True
"""
Set `EXTENDED_EVALUATION` to `True` in order to generate additional plots on validation data.
"""

USE_PRETRAINED_MODEL = True
"""
If `USE_PRETRAINED_MODEL` is `True`, then MAP inference uses provided pretrained weights.
You should not modify MAP training or the CNN architecture before passing the hard baseline.
If you set the constant to `False` (to further experiment),
this solution always performs MAP inference before running your SWAG implementation.
Note that MAP inference can take a long time.
"""


class SWAGInference(object):
    """
    Your implementation of SWA-Gaussian.
    This class is used to run and evaluate your solution.
    You must preserve all methods and signatures of this class.
    However, you can add new methods if you want.

    We provide basic functionality and some helper methods.
    You can pass all baselines by only modifying methods marked with TODO.
    However, we encourage you to skim other methods in order to gain a better understanding of SWAG.
    """

    def __init__(
        self,
        train_xs: torch.Tensor,
        model_dir: pathlib.Path,
        # TODO(1): change inference_mode to InferenceMode.SWAG_DIAGONAL
        # TODO(2): change inference_mode to InferenceMode.SWAG_FULL
        inference_mode: InferenceType = InferenceType.MAP,
        # TODO(2): optionally add/tweak hyperparameters
        swag_training_epochs: int = 30,
        swag_lr: float = 0.045,
        swag_update_interval: int = 1,
        max_rank_deviation_matrix: int = 15,
        num_bma_samples: int = 30,
        device: torch.device = torch.device("cpu"),
    ):
        """
        :param train_xs: Training images (for storage only)
        :param model_dir: Path to directory containing pretrained MAP weights
        :param inference_mode: Control which inference mode (MAP, SWAG-diagonal, full SWAG) to use
        :param swag_training_epochs: Total number of gradient descent epochs for SWAG
        :param swag_lr: Learning rate for SWAG gradient descent
        :param swag_update_interval: Frequency (in epochs) for updating SWAG statistics during gradient descent
        :param max_rank_deviation_matrix: Rank of deviation matrix for full SWAG
        :param num_bma_samples: Number of networks to sample for Bayesian model averaging during prediction
        """

        self.model_dir = model_dir
        self.inference_mode = inference_mode
        self.swag_training_epochs = swag_training_epochs
        self.swag_lr = swag_lr
        self.swag_update_interval = swag_update_interval
        self.max_rank_deviation_matrix = max_rank_deviation_matrix
        self.num_bma_samples = num_bma_samples
        self.device = device

        # Network used to perform SWAG.
        # Note that all operations in this class modify this network IN-PLACE!
        self.network = CNN(in_channels=3, out_classes=6, device=self.device)

        # Store training dataset to recalculate batch normalization statistics during SWAG inference
        self.training_dataset = torch.utils.data.TensorDataset(train_xs.to(self.device))

        # SWAG-diagonal
        # TODO(1): create attributes for SWAG-diagonal
        #  Hint: self._create_weight_copy() creates an all-zero copy of the weights
        #  as a dictionary that maps from weight name to values.
        #  Hint: you never need to consider the full vector of weights,
        #  but can always act on per-layer weights (in the format that _create_weight_copy() returns)
        self.swag_mom1 = self._create_weight_copy()  # All zero copy
        self.swag_mom2 = self._create_weight_copy()
        self.n = 0

        # Full SWAG
        # TODO(2): create attributes for SWAG-full
        #  Hint: check collections.deque
        self.swag_deviations_deque = collections.deque(maxlen=self.max_rank_deviation_matrix)

        # Calibration, prediction, and other attributes
        # TODO(2): create additional attributes, e.g., for calibration
        self._calibration_threshold = None  # this is an example, feel free to be creative

        self.history = {}
        self.last_bma_variances = None

    def update_swag_statistics(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """

        # Create a copy of the current network weights
        copied_params = {name: param.detach() for name, param in self.network.named_parameters()}

        # SWAG-diagonal
        for name, param in copied_params.items():
            # TODO(1): update SWAG-diagonal attributes for weight `name` using `copied_params` and `param`
            # assert (self.swag_mom2[name] >= self.swag_mom1[name]**2).all()
            self.swag_mom1[name] = (self.n / (self.n + 1)) * self.swag_mom1[name] + (1 / (self.n + 1)) * param  # First moment
            self.swag_mom2[name] = (self.n / (self.n + 1)) * self.swag_mom2[name] + (1 / (self.n + 1)) * param ** 2  # Uncentered second moment
            # assert (self.swag_mom2[name] >= self.swag_mom1[name]**2).all()

        # Full SWAG. Only save deviations for the last max_rank_deviation_matrix epochs
        if (self.inference_mode == InferenceType.SWAG_FULL) and (self.n >= self.swag_training_epochs - self.max_rank_deviation_matrix):  
            # TODO(2): update full SWAG attributes for weight `name` using `copied_params` and `param`

            deviations = {}
            for name, param in copied_params.items():
                deviations[name] = param - self.swag_mom1[name]
            self.swag_deviations_deque.append(deviations)
        
        self.n += 1

    def log(self, log_dict: dict):
        for key, value in log_dict.items():
            self.history[key] = self.history.get(key, []) + [value]

    def fit_swag_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag_statistics().
        """

        # We use SGD with momentum and weight decay to perform SWA.
        # See the paper on how weight decay corresponds to a type of prior.
        # Feel free to play around with optimization hyperparameters.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        # TODO(2): Update SWAGScheduler instantiation if you decided to implement a custom schedule.
        #  By default, this scheduler just keeps the initial learning rate given to `optimizer`.
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_training_epochs,
            steps_per_epoch=len(loader),
        )

        # TODO(1): Perform initialization for SWAG fitting
        self.update_swag_statistics()

        self.network.train()
        with tqdm.trange(self.swag_training_epochs, desc="Running gradient descent for SWA") as pbar:
            progress_dict = {}
            for epoch in pbar:
                avg_loss = 0.0
                avg_acc = 0.0
                num_samples = 0
                for i, (batch_images, batch_snow_labels, batch_cloud_labels, batch_labels) in enumerate(loader):
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()
                    lr = lr_scheduler.get_last_lr()[0]
                    progress_dict["lr"] = lr
                    lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (num_samples + batch_images.size(0))
                    batch_acc = torch.mean((predictions.argmax(dim=-1) == batch_labels).float()).item()
                    avg_acc = (batch_images.size(0) * batch_acc + num_samples * avg_acc) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)
                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_acc
                    pbar.set_postfix(progress_dict)
                    self.log({'epoch': epoch, 'batch': i, "loss-batch": batch_loss.item(), 'acc-batch': batch_acc, 'lr': lr})

                # TODO(1): Implement periodic SWAG updates using the attributes defined in __init__
                if epoch % self.swag_update_interval == 0:
                    self.update_swag_statistics()



    def apply_calibration(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate your predictions using a small validation set.
        validation_data contains well-defined and ambiguous samples,
        where you can identify the latter by having label -1.
        """
        if self.inference_mode == InferenceType.MAP:
            # In MAP mode, simply predict argmax and do nothing else
            self._calibration_threshold = 0.0
            return

        # TODO(1): pick a prediction threshold, either constant or adaptive.
        #  The provided value should suffice to pass the easy baseline.
        self._calibration_threshold = 2.0 / 3.0

        # TODO(2): perform additional calibration if desired.
        #  Feel free to remove or change the prediction threshold.
        val_images, val_snow_labels, val_cloud_labels, val_labels = validation_data.tensors
        assert val_images.size() == (140, 3, 60, 60)  # N x C x H x W
        assert val_labels.size() == (140,)
        assert val_snow_labels.size() == (140,)
        assert val_cloud_labels.size() == (140,)

    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        Outputs should be a Nx6 tensor, where N is the number of samples in loader,
        and all rows of the output should sum to 1.
        That is, output row i column j should be your predicted p(y=j | x_i).
        """

        self.network.eval()

        # Perform Bayesian model averaging:
        # Instead of sampling self.num_bma_samples networks (using self.sample_parameters())
        # for each datapoint, you can save time by sampling self.num_bma_samples networks,
        # and perform inference with each network on all samples in loader.
        model_predictions = []
        for i in tqdm.trange(self.num_bma_samples, desc="Performing Bayesian model averaging"):
            # TODO(1): Sample new parameters for self.network from the SWAG approximate posterior
            self.sample_parameters()

            # TODO(1): Perform inference for all samples in `loader` using current model sample,
            #  and add the predictions to model_predictions
            model_predictions.append(self.predict_probabilities_map(loader))

        assert len(model_predictions) == self.num_bma_samples
        assert all(
            isinstance(sample_predictions, torch.Tensor)
            and sample_predictions.dim() == 2  # N x C
            and sample_predictions.size(1) == 6
            for sample_predictions in model_predictions
        )

        # TODO(1): Average predictions from different model samples into bma_probabilities
        bma_probabilities = torch.mean(torch.stack(model_predictions), dim=0)
        bma_variances = torch.var(torch.stack(model_predictions), dim=0)
        self.last_bma_variances = bma_variances

        assert bma_probabilities.dim() == 2 and bma_probabilities.size(1) == 6  # N x C
        return bma_probabilities

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.
        For simplicity, this method directly modifies self.network in-place.
        Hence, after calling this method, self.network corresponds to a new posterior sample.
        """

        K = len(self.swag_deviations_deque)

        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.
        for name, param in self.network.named_parameters():
            # SWAG-diagonal part
            z_diag1 = torch.randn(param.size()).to(self.device)
            # TODO(1): Sample parameter values for SWAG-diagonal
            mean_weights = self.swag_mom1[name]
            std_weights = torch.sqrt(torch.clamp(self.swag_mom2[name] - mean_weights**2, min=0))  # Had problems with negative values?
            assert mean_weights.size() == param.size() and std_weights.size() == param.size()

            # Diagonal part
            sampled_weight = mean_weights + std_weights * z_diag1 * 1/np.sqrt(2)  # FIXME: Check const

            # Full SWAG part
            if self.inference_mode == InferenceType.SWAG_FULL:
                # TODO(2): Sample parameter values for full SWAG
                z_diag2 = torch.randn(K).to(self.device)
                Dhat = torch.stack([self.swag_deviations_deque[i][name] for i in range(K)], dim=-1)  # FIXME: Only do this once
                sampled_weight += torch.matmul(Dhat, z_diag2) * 1/np.sqrt(2*(K-1))

            # Modify weight value in-place; directly changing self.network
            param.data = sampled_weight

        # TODO(1): Don't forget to update batch normalization statistics using self._update_batchnorm_statistics()
        #  in the appropriate place!
        self._update_batchnorm_statistics()

    def predict_labels(self, predicted_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Predict labels in {0, 1, 2, 3, 4, 5} or "don't know" as -1
        based on your model's predicted probabilities.
        The parameter predicted_probabilities is an Nx6 tensor containing predicted probabilities
        as returned by predict_probabilities(...).
        The output should be a N-dimensional long tensor, containing values in {-1, 0, 1, 2, 3, 4, 5}.
        """

        # label_probabilities contains the per-row maximum values in predicted_probabilities,
        # max_likelihood_labels the corresponding column index (equivalent to class).
        label_probabilities, max_likelihood_labels = torch.max(predicted_probabilities, dim=-1)
        num_samples, num_classes = predicted_probabilities.size()
        assert label_probabilities.size() == (num_samples,) and max_likelihood_labels.size() == (num_samples,)

        # A model without uncertainty awareness might simply predict the most likely label per sample:
        # return max_likelihood_labels

        # A bit better: use a threshold to decide whether to return a label or "don't know" (label -1)
        # TODO(2): implement a different decision rule if desired
        return torch.where(
            label_probabilities >= self._calibration_threshold,
            max_likelihood_labels,
            torch.ones_like(max_likelihood_labels) * -1,
        )

    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.network.named_parameters()
        }

    def fit(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Perform full SWAG fitting procedure.
        If `PRETRAINED_WEIGHTS_FILE` is `True`, this method skips the MAP inference part,
        and uses pretrained weights instead.

        Note that MAP inference can take a very long time.
        You should hence only perform MAP inference yourself after passing the hard baseline
        using the given CNN architecture and pretrained weights.
        """
        # MAP inference to obtain initial weights
        PRETRAINED_WEIGHTS_FILE = self.model_dir / "map_weights.pt"
        if USE_PRETRAINED_MODEL:
            self.network.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE, map_location=self.device))
            print("Loaded pretrained MAP weights from", PRETRAINED_WEIGHTS_FILE)
        else:
            self.fit_map_model(loader)

        # SWAG
        if self.inference_mode in (InferenceType.SWAG_DIAGONAL, InferenceType.SWAG_FULL):
            self.fit_swag_model(loader)

    def fit_map_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        MAP inference procedure to obtain initial weights of self.network.
        This is the exact procedure that was used to obtain the pretrained weights we provide.
        """
        map_training_epochs = 140
        initial_learning_rate = 1e-2
        reduced_learning_rate = 1e-3
        start_decay_epoch = 50
        decay_factor = reduced_learning_rate / initial_learning_rate

        # Create optimizer, loss, and a learning rate scheduler that aids convergence
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=initial_learning_rate,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=decay_factor,
                    total_iters=(map_training_epochs - start_decay_epoch) * len(loader),
                ),
            ],
            milestones=[start_decay_epoch * len(loader)],
        )

        # Put network into training mode
        # Batch normalization layers are only updated if the network is in training mode,
        # and are replaced by a moving average if the network is in evaluation mode.
        self.network.train()
        with tqdm.trange(map_training_epochs, desc="Fitting initial MAP weights") as pbar:
            progress_dict = {}
            # Perform the specified number of MAP epochs
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                # Iterate over batches of randomly shuffled training data
                for batch_images, _, _, batch_labels in loader:
                    # Training step
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()

                    # Save learning rate that was used for step, and calculate new one
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    with warnings.catch_warnings():
                        # Suppress annoying warning (that we cannot control) inside PyTorch
                        warnings.simplefilter("ignore")
                        lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)

                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

    def predict_probabilities(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for the given images xs.
        This method returns an NxC float tensor,
        where row i column j corresponds to the probability that y_i is class j.

        This method uses different strategies depending on self.inference_mode.
        """
        self.network = self.network.eval()

        # Create a loader that we can deterministically iterate many times if necessary
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        with torch.no_grad():  # save memory by not tracking gradients
            if self.inference_mode == InferenceType.MAP:
                return self.predict_probabilities_map(loader)
            else:
                return self.predict_probabilities_swag(loader)

    def predict_probabilities_map(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Predict probabilities assuming that self.network is a MAP estimate.
        This simply performs a forward pass for every batch in `loader`,
        concatenates all results, and applies a row-wise softmax.
        """
        all_predictions = []
        for (batch_images,) in loader:
            all_predictions.append(self.network(batch_images))

        all_predictions = torch.cat(all_predictions)
        return torch.softmax(all_predictions, dim=-1)

    def _update_batchnorm_statistics(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.training_dataset.
        We provide this method for you for convenience.
        See the SWAG paper for why this is required.

        Batch normalization usually uses an exponential moving average, controlled by the `momentum` parameter.
        However, we are not training but want the statistics for the full training dataset.
        Hence, setting `momentum` to `None` tracks a cumulative average instead.
        The following code stores original `momentum` values, sets all to `None`,
        and restores the previous hyperparameters after updating batchnorm statistics.
        """

        original_momentum_values = dict()
        for module in self.network.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            original_momentum_values[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            module.reset_running_stats()

        loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.network.train()
        for (batch_images,) in loader:
            self.network(batch_images)
        self.network.eval()

        # Restore old `momentum` hyperparameter values
        for module, momentum in original_momentum_values.items():
            module.momentum = momentum


class CNN(torch.nn.Module):
    """
    Small convolutional neural network used in this task.
    You should not modify this class before passing the hard baseline.

    Note that if you change the architecture of this network,
    you need to re-run MAP inference and cannot use the provided pretrained weights anymore.
    Hence, you need to set `USE_PRETRAINED_INIT = False` at the top of this file.
    """
    def __init__(
        self,
        in_channels: int,
        out_classes: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        ).to(device)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        ).to(device)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        ).to(device)
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=(2, 2)).to(device)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        ).to(device)
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        ).to(device)
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=(2, 2)).to(device)

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
        ).to(device)

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1)).to(device)

        self.linear = torch.nn.Linear(64, out_classes).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)

        # Average features over both spatial dimensions, and remove the now superfluous dimensions
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        log_softmax = self.linear(x)

        return log_softmax