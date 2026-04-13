import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, base, surrogate
from spikingjelly.activation_based import layer


cfg = {
    'vgg9': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P'],
    'cifarnet': [256, 256, 256, 'P', 256, 256, 256, 'P']
}

class VGG(nn.Module):
    def __init__(self, arch, num_cls, img_size, input_dim, act='snn', tau=2., v_threshold=1., drop_rate=0.5, use_freq_weights=False, **kwargs):
        super().__init__()
        
        self.use_freq_weights = use_freq_weights
        if self.use_freq_weights:
            self.freq_weights = nn.Parameter(torch.ones(img_size[0]))
            # 使用高斯分布初始化，均值为1，标准差为0.2
            nn.init.normal_(self.freq_weights, mean=1.0, std=0.2)
        
        assert arch in cfg.keys(), f'Invalid architecture option {arch}'

        # Determine bias flag based on act
        bias_flag = False if act == 'snn' else True

        # Initial dimensions
        h_dim, w_dim = img_size
        in_channel = input_dim

        # Build convolutional layers from cfg
        conv = []
        for x in cfg[arch]:
            if x == 'P':
                conv.append(layer.AvgPool2d(kernel_size=2))
                h_dim //= 2
                w_dim //= 2
            else:
                out_channel = x
                conv.append(layer.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=bias_flag))
                conv.append(layer.BatchNorm2d(out_channel))
                conv.append(neuron.LIFNode(tau=tau, v_threshold=v_threshold) if act == 'snn' else nn.ReLU())
                in_channel = out_channel
        
        # Calculate in_features for the first linear layer
        in_features = h_dim * w_dim * in_channel

        # Build the final sequential model
        self.conv_fc = nn.Sequential(
            *conv,
            layer.Flatten(),
            layer.Dropout(drop_rate),
            nn.Linear(in_features, 384, bias=bias_flag),
            layer.Dropout(drop_rate),
            nn.Linear(384, 192, bias=bias_flag),
            layer.Dropout(drop_rate),
            nn.Linear(192, num_cls, bias=bias_flag),
        )

    def forward(self, x):
        if self.use_freq_weights:
            # x shape: (N, C, H, W), H 是频率维度
             w = torch.sigmoid(self.freq_weights)
             x = x * w.view(1, 1, -1, 1)
        return self.conv_fc(x)


class Fusion(nn.Module):
    def __init__(self, in_dim, pmodel_num, num_cls, act='snn'):
        super().__init__()
        bias_flag = True if act == 'ann' else False

        self.mlp = nn.Sequential(
            layer.Linear(in_dim, 384 * pmodel_num, bias=bias_flag),
            layer.Dropout(0.5),
            layer.Linear(384 * pmodel_num, 192 * pmodel_num, bias=bias_flag),
            layer.Dropout(0.5),
            layer.Linear(192 * pmodel_num, num_cls * pmodel_num, bias=bias_flag),
            layer.Linear(num_cls * pmodel_num, num_cls, bias=bias_flag),
        )

    def forward(self, x):
        return self.mlp(x)