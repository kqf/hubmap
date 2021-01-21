import torch


class BCEWithLogitsLossPadding(torch.nn.Module):
    def __init__(self, padding=0):
        super().__init__()
        self.padding = padding

    def forward(self, input, target):
        x = input.squeeze_(dim=1)
        _, h, w = x.shape
        x = x[:, self.padding:h - self.padding, self.padding:w - self.padding]
        y = target.squeeze_(dim=1).float()
        y = y[:, self.padding:h - self.padding, self.padding:w - self.padding]
        return torch.nn.functional.binary_cross_entropy_with_logits(x, y)
