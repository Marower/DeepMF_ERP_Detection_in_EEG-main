import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            #Conv1d(in_channels, out_channels, kernel_size,)
            nn.Conv1d(4, 6, 200, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(6, 6, 50, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(6, 6, 50, stride=1, padding=1),
            nn.Sigmoid(),
  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(6, 6, 50, stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(6, 6, 50, stride=1, padding=1),
            nn.ConvTranspose1d(6, 4, 200, stride=1,padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
