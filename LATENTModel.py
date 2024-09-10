# Several basic machine learning models
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math

class LatentModule(nn.Module):
    def __init__(self, num_bits_whole, num_bits_fraction, epsilon, flatten_hidden_size,
                 alpha, device):
        super().__init__()
        self.num_bits_whole = num_bits_whole
        self.num_bits_fraction = num_bits_fraction
        self.flatten_hidden_size = flatten_hidden_size
        self.epsilon = epsilon
        self.total_bits = self.num_bits_fraction + self.num_bits_whole + 1
        self.sensitivity = self.total_bits * self.flatten_hidden_size
        self.alpha = alpha
        self.device = device

        # UER
        self.even_1to1 = self.alpha / (1 + self.alpha) # p
        self.even_0to1 = 1 / (1 + self.alpha * math.exp(self.epsilon / self.sensitivity)) # q
        self.even_1to0 = 1 - self.even_1to1 # 1-p
        self.even_0to0 = 1 - self.even_0to1 # 1-q
        self.odd_1to1 = 1 / (1 + self.alpha**3) # p
        self.odd_0to1 = 1 / (1 + self.alpha * math.exp(self.epsilon / self.sensitivity)) # q
        self.odd_1to0 = 1 - self.odd_1to1 # 1-p
        self.odd_0to0 = 1 - self.odd_0to1 # 1-q


    def _my_convert(self, x, num_bits, mode='right'):
        if mode == 'right':
            result = torch.zeros(size=(x.shape[0], self.num_bits_whole), dtype=torch.float32)
            for idx, value in enumerate(x):
                res = [math.floor(math.fabs(value) * (2 ** (-i))) % 2 for i in range(0, num_bits, 1)]
                result[idx] = torch.tensor(res, dtype=torch.float32)
        elif mode == 'left':
            result = torch.zeros(size=(x.shape[0], self.num_bits_fraction), dtype=torch.float32)
            for idx, value in enumerate(x):
                res = [math.floor(math.fabs(value) * (2 ** i)) % 2 for i in range(1, num_bits + 1, 1)]
                result[idx] = torch.tensor(res, dtype=torch.float32)
        else:
            raise NotImplementedError

        return result




    def _convert_to_binary(self, x):
        """
        x: (batch_size, hidden_size)
        binary_x: (batch_size, (hidden_size * self.total_bits))
        """

        binary_sign = torch.less(x.detach(), 0).type(torch.float32)
        binary_sign = binary_sign.unsqueeze(-1)

        batch_size, hidden_size = x.shape[0], x.shape[1]
        x = x.reshape(-1).unsqueeze(-1)
        binary_whole = self._my_convert(x, self.num_bits_whole, mode='right').to(self.device)

        binary_frac = self._my_convert(x, self.num_bits_fraction, mode='left').to(self.device)

        binary_whole = binary_whole.reshape(batch_size, hidden_size, -1)
        binary_frac = binary_frac.reshape(batch_size, hidden_size, -1)

        # merge the results
        binary_x = torch.cat((binary_sign, binary_whole, binary_frac), dim=-1)
        binary_x = binary_x.reshape(batch_size, -1)
        return binary_x

    def _randomize(self, x):
        """
        UER algorithm
        x: (batch_size, binary_length)
        """
        B, L = x.shape[0], x.shape[1]
        random_values = np.random.uniform(low=0., high=1., size=(B, L))
        random_values = torch.tensor(random_values, device=self.device)
        odd_lists = torch.arange(1, x.size(1), 2)
        even_lists = torch.arange(0, x.size(1), 2)
        odd_value = x[:, odd_lists]
        even_value = x[:, even_lists]
        new_odd_value = torch.where(
            odd_value == 1.,
            torch.where(random_values[:, odd_lists] <= self.odd_1to1, 1., 0.),
            torch.where(random_values[:, odd_lists] <= self.odd_0to1, 1., 0.),
        ).tolist()
        new_even_value = torch.where(
            even_value == 1.,
            torch.where(random_values[:, even_lists] <= self.even_1to1, 1., 0.),
            torch.where(random_values[:, even_lists] <= self.even_0to1, 1., 0.),
        ).tolist()

        # concatenate the final result
        # https://stackoverflow.com/a/18041868/21254589
        n = 2 * max([len(new_even_value[0]), len(new_odd_value[0])])
        result = [n * [None] for _ in range(B)]
        for res, new_even, new_odd in zip(result, new_even_value, new_odd_value):
            res[0:2*len(new_even):2] = new_even
            res[1:2*len(new_odd):2] = new_odd
            res = [x for x in res if x != None]
        return torch.tensor(result, device=self.device)

    def forward(self, x):
        normalized_x = (x - torch.mean(x, dim=1, keepdim=True)) / torch.sqrt(torch.var(x, dim=1, keepdim=True))
        binary_x = self._convert_to_binary(normalized_x)
        randomized_x = self._randomize(binary_x)

        return randomized_x


class LATENTMnistCNN(nn.Module):
    def __init__(self, data_in, data_out, n, m, epsilon, alpha, device):
        super(LATENTMnistCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.LATENT = LatentModule(
            num_bits_whole=n,
            num_bits_fraction=m,
            epsilon=epsilon,
            flatten_hidden_size=1568,
            alpha=alpha,
            device=device,
        )
        self.fc = nn.Linear(1568 * self.LATENT.total_bits, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1) # (128, 1568)
        t1 = time.time()
        x = self.LATENT(x)
        t2 = time.time()
        # print(f"cost: time for one batch: {t2-t1:.4f}")
        x = self.fc(x)
        return x


class LATENTCifarCNN(nn.Module):
    def __init__(self, data_in, data_out, n, m, epsilon, alpha, device):
        super(LATENTCifarCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.LATENT = LatentModule(
            num_bits_whole=n,
            num_bits_fraction=m,
            epsilon=epsilon,
            flatten_hidden_size=100,
            alpha=alpha,
            device=device,
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = x.view(-1, 64 * 4 * 4)
        x = torch.flatten(x, 1)
        x = self.LATENT(x)
        x = self.fc(x)
        return x

