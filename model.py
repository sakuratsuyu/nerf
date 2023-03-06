import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, layer = 8, channel = 256, skips = [4], input_ch = 3, input_ch_views = 3) -> None:
        '''
            Args:
                layer: int. number of layers.
                channel: int. number of channels.
                input_ch: int. number of position input channel.
                input_ch_views: int, number of direction input channel.
                skips: int. index of the layer that adds position encoding again.
                use_viewdirs: boolean. whether to use 5D input.
        '''

        super(NeRF, self).__init__()

        self.layer = layer
        self.channel = channel
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips

        self.pos_linears = nn.ModuleList(
            [nn.Linear(input_ch, channel)]+
            [nn.Linear(channel, channel) if i not in skips else nn.Linear(channel + input_ch, channel) for i in range(layer - 1)]
        )

        self.view_linear = nn.Linear(input_ch_views + channel, channel // 2)
        self.feature_linear = nn.Linear(channel, channel)
        self.alpha_linear = nn.Linear(channel, 1)
        self.rgb_linear = nn.Linear(channel // 2, 3)

    def forward(self, input_pos:torch.Tensor, input_views:torch.Tensor):
        '''
            From (pos + dir) to (RGB + sigma).

            Args:
                input_pos: <class 'torch.Tensor'>. (batch, 3). batch input of position(3D)
                input_views: <class 'torch.Tensor'>. (batch, 3). batch input of direction(3D).

            Returns:
                output: <class 'numpy.ndarray'>. (batch, 3 + 1). batch output of RGB(3D) and sigma(1D).
        '''

        temp = input_pos
        for i, _ in enumerate(self.pos_linears):
            temp = self.pos_linears[i](temp)
            temp = F.relu(temp)

            if i in self.skips:
                temp = torch.cat([input_pos, temp], dim=-1)

        feature = self.feature_linear(temp)
        alpha = self.alpha_linear(temp)

        temp = torch.cat([feature, input_views], dim=-1)
        temp = self.view_linear(temp)
        temp = F.relu(temp)

        temp = self.rgb_linear(temp)
        rgb = torch.sigmoid(temp)

        return torch.cat([rgb, alpha], dim=-1)

class Embedder:
    def __init__(self, input_dim, max_freq_log, num_freq, include_input):
        '''
            Args:
                input_dim: int. input dimension.
                max_freq_log: int. log 2 of embedded maximum frequency.
                num_freq: int. number of embedded frequencies.
                include_input: boolean. whether to include input for Embedder.
        '''

        fns = [torch.cos, torch.sin]
        embed_fns = []

        if include_input:
            embed_fns.append(lambda x : x)

        max_freq = max_freq_log
        N_freq = num_freq
        freq_bds = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freq)

        for freq in freq_bds:
            for fn in fns:
                def func(freq, fn):
                    def fun(p):
                        return fn(freq * p)
                    return fun
                embed_fns.append(func(freq, fn))

        self.output_dim = len(embed_fns) * input_dim
        self.embed_fns = embed_fns

    def __call__(self, input):
        return torch.cat([fn(input) for fn in self.embed_fns], dim=-1)
