from pytorch_lightning.callbacks import EarlyStopping


class TrainEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)

