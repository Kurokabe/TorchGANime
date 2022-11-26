import os

os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import copy
from pprint import pprint

from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch
import click
import pytorch_lightning as pl
import ray
import ray.tune as tune
from omegaconf import OmegaConf

# from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

# from cli import RayTuneCli
from torchganime.data.image import ImageData
from torchganime.models.vqgan import VQGAN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"


def train_model(config, initial_config, callbacks):
    current_config = copy.deepcopy(initial_config)
    for key, value in config.items():
        OmegaConf.update(current_config, key, value)

    print("loading model...")
    model = VQGAN(**current_config["model"])
    print("loading data...")
    data = ImageData(**current_config["data"])
    trainer = pl.Trainer(
        **current_config["trainer"],
        logger=TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), "raytune"), name="", version="."
        ),
        enable_progress_bar=False,
        callbacks=callbacks,
        devices=[0],
        # terminate_on_nan=True,
    )
    print("training...")
    trainer.fit(model, data.train_dataloader(), data.val_dataloader())


def tune_model(initial_config, max_concurrent, num_samples):
    # hyperparameters to search
    hparams = {
        "model.learning_rate": tune.qloguniform(1e-5, 1e-3, 5e-6),
        # "model.autoencoder_config.embed_dim": tune.choice([32, 64, 128, 256]),
        # "model.autoencoder_config.n_embed": tune.choice([1024, 50257]),
        # "model.autoencoder_config.channels": tune.choice([32, 64, 128, 256]),
        # "model.autoencoder_config.z_channels": tune.choice([32, 64, 128, 256]),
        # "model.autoencoder_config.ch_mult": tune.choice(
        #     [
        #         [1, 1, 2, 2, 4],
        #         [1, 2, 2, 4, 4],
        #         [1, 1, 2, 4, 4],
        #         [1, 1, 2, 2, 2],
        #         # [1, 2, 4, 8, 8],
        #         # [1, 2, 4, 8],
        #         # [1, 2, 2, 4],
        #     ]
        # ),attn_resolutions
        "model.autoencoder_config.attn_resolutions": tune.choice([[], [16]]),
        "model.autoencoder_config.dropout": tune.quniform(0.0, 0.3, 0.05),
        "model.autoencoder_config.resamp_with_conv": tune.choice([True, False]),
        "model.loss_config.disc_use_actnorm": tune.choice([True, False]),
    }

    # metrics to track: keys are displayed names and
    # values are corresponding labels defined in LightningModule
    metrics = {"loss": "val/rec_loss_epoch"}

    # scheduler
    scheduler = AsyncHyperBandScheduler(max_t=50, grace_period=10)

    # progress reporter
    reporter = CLIReporter(
        parameter_columns={p: p.split(".")[-1] for p in hparams.keys()},
        metric_columns=list(metrics.keys()),
    )

    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    resources_per_trial = {"cpu": 1, "gpu": 1}

    # main analysis
    trainable_function = tune.with_parameters(
        train_model, initial_config=initial_config, callbacks=callbacks
    )

    result = tune.run(
        trainable_function,
        resources_per_trial=resources_per_trial,
        config=hparams,
        search_alg=ConcurrencyLimiter(OptunaSearch(), max_concurrent=max_concurrent),
        scheduler=scheduler,
        progress_reporter=reporter,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        max_concurrent_trials=max_concurrent,
        reuse_actors=False,
    )

    best_trial = result.get_best_trial("loss", "min", "all")
    print("Best hyperparameters found were:")
    pprint(best_trial.config)
    print("Corresponding metrics are:")
    pprint({metric: best_trial.last_result[metric] for metric in metrics.keys()})


@click.command()
@click.option("-c", "--config", type=str, default=None)
def tune_main(config):

    ray.init(num_cpus=10, num_gpus=8)
    conf = OmegaConf.load(config)

    tune_model(initial_config=conf, max_concurrent=8, num_samples=100)


if __name__ == "__main__":
    tune_main()

# import os

# os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
# import math

# import torch
# import pytorch_lightning as pl
# from filelock import FileLock
# from torch.utils.data import DataLoader, random_split
# from torch.nn import functional as F
# from torchvision.datasets import MNIST
# from torchvision import transforms

# from pytorch_lightning.loggers import TensorBoardLogger
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
# from ray.tune.integration.pytorch_lightning import (
#     TuneReportCallback,
#     TuneReportCheckpointCallback,
# )


# class LightningMNISTClassifier(pl.LightningModule):
#     """
#     This has been adapted from
#     https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
#     """

#     def __init__(self, config, data_dir=None):
#         super(LightningMNISTClassifier, self).__init__()

#         self.data_dir = data_dir or os.getcwd()

#         self.layer_1_size = config["layer_1_size"]
#         self.layer_2_size = config["layer_2_size"]
#         self.lr = config["lr"]
#         self.batch_size = config["batch_size"]

#         mnist images are (1, 28, 28) (channels, width, height)
#         self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
#         self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
#         self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)

#     def forward(self, x):
#         batch_size, channels, width, height = x.size()
#         x = x.view(batch_size, -1)

#         x = self.layer_1(x)
#         x = torch.relu(x)

#         x = self.layer_2(x)
#         x = torch.relu(x)

#         x = self.layer_3(x)
#         x = torch.log_softmax(x, dim=1)

#         return x

#     def cross_entropy_loss(self, logits, labels):
#         return F.nll_loss(logits, labels)

#     def accuracy(self, logits, labels):
#         _, predicted = torch.max(logits.data, 1)
#         correct = (predicted == labels).sum().item()
#         accuracy = correct / len(labels)
#         return torch.tensor(accuracy)

#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         accuracy = self.accuracy(logits, y)

#         self.log("ptl/train_loss", loss)
#         self.log("ptl/train_accuracy", accuracy)
#         return loss

#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         accuracy = self.accuracy(logits, y)
#         return {"val_loss": loss, "val_accuracy": accuracy}

#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#         avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
#         self.log("ptl/val_loss", avg_loss)
#         self.log("ptl/val_accuracy", avg_acc)

#     @staticmethod
#     def download_data(data_dir):
#         transform = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#         )
#         with FileLock(os.path.expanduser("~/.data.lock")):
#             return MNIST(data_dir, train=True, download=True, transform=transform)

#     def prepare_data(self):
#         mnist_train = self.download_data(self.data_dir)

#         self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

#     def train_dataloader(self):
#         return DataLoader(self.mnist_train, batch_size=int(self.batch_size))

#     def val_dataloader(self):
#         return DataLoader(self.mnist_val, batch_size=int(self.batch_size))

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         return optimizer


# def train_mnist_tune(config, num_epochs=10, num_gpus=0, data_dir="~/data"):
#     data_dir = os.path.expanduser(data_dir)
#     model = LightningMNISTClassifier(config, data_dir)
#     trainer = pl.Trainer(
#         max_epochs=num_epochs,
#         If fractional GPUs passed in, convert to int.
#         devices=math.ceil(num_gpus),
#         logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
#         enable_progress_bar=False,
#         callbacks=[
#             TuneReportCallback(
#                 {"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"},
#                 on="validation_end",
#             )
#         ],
#         deterministic=False,
#         accelerator="gpu",
#     )
#     trainer.fit(model)


# def tune_mnist_asha(num_samples=10, num_epochs=10, gpus_per_trial=0, data_dir="~/data"):
#     config = {
#         "layer_1_size": tune.choice([32, 64, 128]),
#         "layer_2_size": tune.choice([64, 128, 256]),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([32, 64, 128]),
#     }

#     scheduler = ASHAScheduler(max_t=num_epochs, grace_period=10, reduction_factor=2)

#     reporter = CLIReporter(
#         parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
#         metric_columns=["loss", "mean_accuracy", "training_iteration"],
#     )

#     train_fn_with_parameters = tune.with_parameters(
#         train_mnist_tune,
#         num_epochs=num_epochs,
#         num_gpus=gpus_per_trial,
#         data_dir=data_dir,
#     )
#     resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

#     analysis = tune.run(
#         train_fn_with_parameters,
#         resources_per_trial=resources_per_trial,
#         metric="loss",
#         mode="min",
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         name="tune_mnist_asha",
#     )

#     print("Best hyperparameters found were: ", analysis.best_config)


# if __name__ == "__main__":
#     tune_mnist_asha(
#         num_samples=10,
#         num_epochs=100,
#         gpus_per_trial=1,
#     )
