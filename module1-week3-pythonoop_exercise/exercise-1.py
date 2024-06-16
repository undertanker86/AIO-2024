import torch.nn as nn
import torch


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.calcu_softmax(x)

    def calcu_softmax(self, x):
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x, dim=0)


class softmax_stable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.calcu_softmax_stable(x)

    def calcu_softmax_stable(self, x):
        c = torch.max(x)
        exp_x_shifted = torch.exp(x - c)
        return exp_x_shifted / torch.sum(exp_x_shifted, dim=0)


if __name__ == "__main__":

    x = torch.tensor([1, 2, 3])

    softmax = Softmax()
    print(softmax(x))

    softmax_stable = softmax_stable()
    print(softmax_stable(x))
