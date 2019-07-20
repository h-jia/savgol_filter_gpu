# Author: h-jia

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
torch.manual_seed(0)


def coeff(order=2, width=5):
    z = torch.arange(-(width - 1) / 2, (width + 1) / 2).reshape(-1, 1)
    J = torch.cat([torch.pow(z, n) for n in range(order + 1)], 1)
    JTJ = torch.matmul(J.t(), J)
    iJTJ = torch.inverse(JTJ)
    C = torch.matmul(iJTJ, J.t())
    return C


def savgol(data, order=2, width=5, h=2):
    savgolcoeff = coeff(order, width)
    coeff = savgolcoeff[0, :]
    pad_length = h * (width - 1) // 2
    half_window = pad_length
    data_pad = F.pad(data, (pad_length, pad_length), 'constant', 0)
    data_len = len(data)
    data_pad_len = len(data_pad)
    new_data = torch.zeros(data_len)
    for i in range(pad_length, data_pad_len - pad_length):
        data_window = data_pad[i - half_window:i + half_window + 1]
        data_smooth = data_window[[h * n for n in range(width)]]
        new_data[i - pad_length] = torch.sum(torch.mul(data_smooth, coeff))
    return new_data


s = torch.rand(150)
plt.plot(s.numpy(), 'b')
snew3 = savgol(s, 3, 13, 1)
plt.plot(snew3.numpy(), 'y')
plt.show()
