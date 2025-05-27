import torch
from torchmetrics import Metric

class MeanSquaredError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        sum_squared_error = torch.sum((preds - target) ** 2)
        total = target.numel()

        self.sum_squared_error += sum_squared_error
        self.total += total

    def compute(self):
        return self.sum_squared_error / self.total
