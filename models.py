import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn


class my_fnn_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(my_fnn_model, self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)

        )

    def forward(self, x):
        x=x.reshape(-1,224*224*3)
        x = self.layer1(x)
        return x


class my_cnn_model(nn.Modul):
    def __init__(self):
        super(my_cnn_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(44944, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(BasicBlock, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride, padding),
            nn.BatchNorm2d(out_channels)
        )

        # Identity (shortcut) path
        self.identity_path = nn.Sequential()
        if in_channels != out_channels:
            self.identity_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding),
                nn.BatchNorm2d(out_channels)
            )

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity_path(x)
        out = self.main_path(x) + identity
        return self.relu(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        # Define the number of intermediate channels
        inter_planes = in_planes // ratio

        # Shared transformation layers for both avg_pool and max_pool
        self.shared_transform = nn.Sequential(
            nn.Conv2d(in_planes, inter_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_planes, in_planes, 1, bias=False)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)

        # Apply the same transformation to both avg_pooled and max_pooled features
        avg_out = self.shared_transform(avg_pooled)
        max_out = self.shared_transform(max_pooled)

        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.spatial_attention(x)


class My_resnet(nn.Module):
    def __init__(self, in_channel, hidden_channels, out_channel,model='resnet'):
        super(My_resnet, self).__init__()
        self.model=model
        # Initial layers
        self.initial_layers = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            BasicBlock(hidden_channels[0][0], hidden_channels[0][1], hidden_channels[0][2], hidden_channels[0][3]),
            BasicBlock(hidden_channels[1][0], hidden_channels[1][1], hidden_channels[1][2], hidden_channels[1][3]),
            BasicBlock(hidden_channels[2][0], hidden_channels[2][1], hidden_channels[2][2], hidden_channels[2][3]),
            BasicBlock(hidden_channels[3][0], hidden_channels[3][1], hidden_channels[3][2], hidden_channels[3][3])
        )

        # Attention mechanisms
        self.channel_attention = ChannelAttention(64)
        self.spatial_attention = SpatialAttention()

        # Final layers
        self.final_layers = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(hidden_channels[3][1] * 1 * 1, out_channel),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        out = self.initial_layers(x)
        out = self.res_blocks(out)
        if self.model!='resnet':
            out = self.channel_attention(out) * out
            out = self.spatial_attention(out) * out
        out = self.final_layers(out)
        return out




