import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError

import os 
from datetime import datetime
import matplotlib.pyplot as plt

from ..train.losses import CategoricalCrossEntropyLoss, FocalLoss
from ..train.custom_iou import calculate_iou


class GenericLightningNetwork(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=1e-3):
        super(GenericLightningNetwork, self).__init__()
        self.lr = learning_rate
        self.model = model
        self._initialize_metrics(num_classes)



    def _initialize_metrics(self, num_classes):
        # Metrics
        if num_classes > 2:
            self.loss_fn = nn.CrossEntropyLoss()
            self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
            self.f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes)
            self.mcc = torchmetrics.classification.MulticlassMatthewsCorrCoef(num_classes=num_classes)
            self.conf_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_classes)
            self.conf_matrix_pred = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=num_classes)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            self.accuracy = torchmetrics.classification.BinaryAccuracy()
            self.f1_score = torchmetrics.classification.BinaryF1Score()
            self.mcc = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
            self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix()
            self.conf_matrix_pred = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        _, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, dim=1), y)
        self.f1_score.update(torch.argmax(scores, dim=1), y)
        self.mcc.update(torch.argmax(scores, dim=1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        import time
        x, y = batch

        start_time = time.time()
        loss, scores, y = self._common_step(batch, batch_idx)
        if x.is_cuda:
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        fps = x.shape[0] / elapsed_time if elapsed_time > 0 else 0.0

        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.conf_matrix.update(torch.argmax(scores, dim=1), y)
        self.conf_matrix.compute()
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_mcc': mcc.float(),
            'fps': fps,
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss


    def on_test_end(self):
        self.conf_matrix.plot()  # to plot and save confusion matrix
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(rf"./logs/tb_logs", exist_ok=True)
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_{current_datetime}.png")
        plt.show()

    def _common_step(self, batch, _):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, _):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)
        f1_score = self.f1_score(preds, y)
        mcc = self.mcc(preds, y)
        self.conf_matrix_pred.update(preds, y)
        self.conf_matrix_pred.compute()

        print(f"Accuracy: {accuracy:.3f}")   
        print(f"F1-score: {f1_score:.3f}")
        print(f"MCC: {mcc:.3f} ")
        return preds

    """
    def on_predict_end(self):
        fig_, ax_ = self.conf_matrix_pred.plot()
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_predictions_{current_datetime}.png")
        plt.show()  # test block=False
    """

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer



class GenericLightningSegmentationNetwork(pl.LightningModule):
    """
    GenericLightningSegmentationNetwork is a PyTorch Lightning module designed for segmentation tasks. 
    It wraps a given model and provides training, validation, testing, and prediction steps, 
    along with logging for loss, mean squared error (MSE), and intersection over union (IoU).
    Attributes:
        model (torch.nn.Module): The segmentation model to be trained and evaluated.
        learning_rate (float): The learning rate for the optimizer. Default is 1e-3.
        loss_fn (callable): The loss function used for training. Default is FocalLoss.
        mse (torchmetrics.Metric): Metric to compute mean squared error.
        iou (callable): Function to calculate intersection over union (IoU).
    Methods:
        forward(x):
            Performs a forward pass through the model.
        _common_step(batch, batch_idx):
            Computes the loss, MSE, and IoU for a given batch. Used internally by training, validation, and test steps.
        training_step(batch, batch_idx):
            Defines the training step, computes metrics, and logs them.
        validation_step(batch, batch_idx):
            Defines the validation step, computes metrics, and logs them.
        test_step(batch, batch_idx):
            Defines the test step, computes metrics, and logs them.
        predict_step(batch, batch_idx, dataloader_idx=0):
            Defines the prediction step, returning the model's output for a given batch.
        configure_optimizers():
            Configures the optimizer for training. Uses Adam optimizer with the specified learning rate.
    """
    def __init__(self, model, learning_rate=1e-3):
        super(GenericLightningSegmentationNetwork, self).__init__()
        self.lr = learning_rate
        self.model = model
        
        #self.loss_fn = CategoricalCrossEntropyLoss()
        self.loss_fn = FocalLoss()
        self.mse = MeanSquaredError()
        self.iou = calculate_iou

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        mse = self.mse(logits, y)
        iou = self.iou(logits, y).mean()  # Compute mean IoU for logging
        return loss, mse, iou

    def training_step(self, batch, batch_idx):
        loss, mse, iou = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        self.log('train_mse', mse)
        self.log('train_iou', iou)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse, iou = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_mse', mse)
        self.log('val_iou', iou)
        return loss

    def test_step(self, batch, batch_idx):
        loss, mse, iou = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_mse', mse)
        self.log('test_iou', iou)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        logits = self(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class GenericLightningNetwork_Custom(pl.LightningModule):
    def __init__(self, parsed_layers, model_parameters, input_channels, num_classes, learning_rate=1e-3):
        super(GenericLightningNetwork_Custom, self).__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            model_parameters=model_parameters,
            input_channels=input_channels,
            num_classes=num_classes,
        )
        self.class_weights = None  # Initialize with a default value

        # Metrics
        self.loss_fn = ce_loss  # Use custom loss function
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1_score = torchmetrics.classification.BinaryF1Score()
        self.mcc = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
        self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix()
        self.conf_matrix_pred = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # Ensure the datamodule is attached and has class_weights
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'class_weights'):
            self.class_weights = self.trainer.datamodule.class_weights.to(self.device)
            print(f"GenericLightningNetwork class_weights set: {self.class_weights}")
        else:
            print("GenericLightningNetwork class_weights NOT set")

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, dim=1), y)
        self.f1_score.update(torch.argmax(scores, dim=1), y)
        self.mcc.update(torch.argmax(scores, dim=1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.conf_matrix.update(torch.argmax(scores, dim=1), y)
        self.conf_matrix.compute()
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def on_test_end(self):
        fig_, ax_ = self.conf_matrix.plot()  # to plot and save confusion matrix
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_{current_datetime}.png")
        # plt.show()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        if self.class_weights is not None:
            loss = self.loss_fn(logits=scores, targets=y, weight=self.class_weights, use_hard_labels=True)
        else:
            loss = self.loss_fn(logits=scores, targets=y, use_hard_labels=True)

        loss = loss.mean()
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)
        f1_score = self.f1_score(preds, y)
        mcc = self.mcc(preds, y)
        self.conf_matrix_pred.update(preds, y)
        self.conf_matrix_pred.compute()

        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1-score: {f1_score:.3f}")
        print(f"MCC: {mcc:.3f} ")
        return preds



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer


def ce_loss(logits, targets, weight=None, use_hard_labels=True, reduction="none"):
    """
    Wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        weight: weights for loss if hard labels are used.
        use_hard_labels: If True, targets have [Batch size] shape with int values.
                         If False, the target is vector. Default to True.
    """
    if use_hard_labels:
        if weight is not None:
            return F.cross_entropy(
                logits, targets.long(), weight=weight, reduction=reduction
            )
        else:
            return F.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

