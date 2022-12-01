import torch
from torch.nn import ReLU
from torch.nn import Conv1d
from torch.nn import Sequential as Seq
from torch.nn import Linear as Lin
from torch.nn import Upsample


def conv1x3(channels, stride=1):
    return Seq(*[
        Seq(
            Conv1d(channels[i - 1],
                   channels[i],
                   3,
                   stride=stride,
                   padding=1,
                   padding_mode='replicate'), ReLU())
        for i in range(1, len(channels))
    ])


def conv1x3_up(channels, scale=2):
    return Seq(*[
        Seq(Upsample(
            scale_factor=scale), conv1x3([channels[i - 1], channels[i]]))
        for i in range(1, len(channels))
    ])


class FINTA(torch.nn.Module):
    def __init__(self, input_size, embedding_size=32):
        super(FINTA, self).__init__()
        self.emb_size = embedding_size
        self.in_size = input_size

        self.encoder = Seq(conv1x3([input_size, 32]),
                           conv1x3([32, 64, 128, 256, 512, 1024], stride=2))

        self.compress = Lin(1024 * 8, embedding_size)
        self.decompress = Seq(Lin(embedding_size, 1024 * 8), ReLU())

        self.decoder = Seq(
            conv1x3_up([1024, 512, 256, 128, 64, 32]),
            Conv1d(32,
                   input_size,
                   3,
                   stride=1,
                   padding=1,
                   padding_mode='replicate'))

    def forward(self, data):
        # data has shape (bs * n_pts, in_size)
        bs = data.batch.max() + 1
        x = data.x.view(bs, -1, self.in_size)
        x = x.permute(0, 2, 1).contiguous()  # shape = (bs, in_size, n_pts)
        x = self.encoder(x)
        # x has shape (bs, 1024, 8)

        emb = self.compress(x.view(bs, -1))
        self.emb = emb  # shape = (bs, 32)
        x = self.decompress(emb).view(bs, x.size(1), -1)
        # x has shape (bs, 1024, 8)

        x = self.decoder(x)
        # x has shape (bs, in_size, n_pts)
        x = x.permute(0, 2, 1).contiguous().view(-1, self.in_size)
        # x has shape (bs * n_pts, in_size)
        return x