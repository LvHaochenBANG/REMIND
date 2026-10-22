import torch

class Metrics:
    # ====== 确定性指标 ======
    @staticmethod
    def mae(y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true)).item()

    @staticmethod
    def rmse(y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

    @staticmethod
    def mape(y_pred, y_true, mask_val=1.0):
        mask = y_true >= mask_val
        if torch.sum(mask).item() == 0:
            return float('nan')
        return (torch.mean(torch.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100).item()
