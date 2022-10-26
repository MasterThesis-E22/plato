"""
Base class for trainers.
"""

from abc import ABC, abstractmethod
import os

from plato.config import Config


class Trainer(ABC):
    """Base class for all the trainers."""

    def __init__(self):
        self.device = Config().device()
        self.client_id = 0

    def set_client_id(self, client_id):
        """Setting the client ID"""
        self.client_id = client_id

    @abstractmethod
    def save_model(self, filename=None, location=None):
        """Saving the model to a file."""
        raise TypeError("save_model() not implemented.")

    @abstractmethod
    def load_model(self, filename=None, location=None):
        """Loading pre-trained model weights from a file."""
        raise TypeError("load_model() not implemented.")

    @staticmethod
    def save_auroc(auroc, filename=None):
        """Saving the test auroc to a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if filename is not None:
            auroc_path = f"{model_path}/{filename}"
        else:
            auroc_path = f"{model_path}/{model_name}.auroc"

        with open(auroc_path, "w", encoding="utf-8") as file:
            file.write(str(auroc))
    
    @staticmethod
    def save_accuracy(accuracy, filename=None):
        """Saving the test accuracy to a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.accuracy"

        with open(accuracy_path, "w", encoding="utf-8") as file:
            file.write(str(accuracy))

    @staticmethod
    def save_loss(loss, filename=None):
        """Saving the test accuracy to a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.testloss"

        with open(accuracy_path, "w", encoding="utf-8") as file:
            file.write(str(loss))

    @staticmethod
    def save_precision(precision, filename=None):
        """Saving the test accuracy to a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.precision"

        with open(accuracy_path, "w", encoding="utf-8") as file:
            file.write(str(precision))

    @staticmethod
    def save_recall(recall, filename=None):
        """Saving the test accuracy to a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.recall"

        with open(accuracy_path, "w", encoding="utf-8") as file:
            file.write(str(recall))
            
    @staticmethod
    def load_auroc(filename=None):
        """Loading the test auroc from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if filename is not None:
            auroc = f"{model_path}/{filename}"
        else:
            auroc = f"{model_path}/{model_name}.auroc"

        with open(auroc, "r", encoding="utf-8") as file:
            auroc = float(file.read())

        return auroc        

    @staticmethod
    def load_accuracy(filename=None):
        """Loading the test accuracy from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.accuracy"

        with open(accuracy_path, "r", encoding="utf-8") as file:
            accuracy = float(file.read())

        return accuracy

    @staticmethod
    def load_loss(filename=None):
        """Loading the loss from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.loss"

        with open(accuracy_path, "r", encoding="utf-8") as file:
            loss = float(file.read())

        return loss

    @staticmethod
    def load_precision(filename=None):
        """Loading the loss from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.precision"

        with open(accuracy_path, "r", encoding="utf-8") as file:
            precision = float(file.read())
        return precision

    @staticmethod
    def load_recall(filename=None):
        """Loading the loss from a file."""
        model_path = Config().params["model_path"]
        model_name = Config().trainer.model_name

        if filename is not None:
            accuracy_path = f"{model_path}/{filename}"
        else:
            accuracy_path = f"{model_path}/{model_name}.recall"

        with open(accuracy_path, "r", encoding="utf-8") as file:
            recall = float(file.read())
        return recall

    def pause_training(self):
        """Remove files of running trainers."""
        if hasattr(Config().trainer, "max_concurrency"):
            model_name = Config().trainer.model_name
            model_path = Config().params["model_path"]
            model_file = f"{model_path}/{model_name}_{self.client_id}_{Config().params['run_id']}.pth"
            accuracy_file = f"{model_path}/{model_name}_{self.client_id}_{Config().params['run_id']}.acc"

            if os.path.exists(model_file):
                os.remove(model_file)
                os.remove(model_file + ".pkl")

            if os.path.exists(accuracy_file):
                os.remove(accuracy_file)

    @abstractmethod
    def train(self, trainset, sampler, **kwargs) -> float:
        """The main training loop in a federated learning workload.

        Arguments:
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.

        Returns:
        float: The training time.
        """

    @abstractmethod
    def test(self, testset, sampler=None, **kwargs) -> float:
        """Testing the model using the provided test dataset.

        Arguments:
        testset: The test dataset.
        sampler: The sampler that extracts a partition of the test dataset.
        """
