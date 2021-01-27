import torch
from typing import List
import segmentation_models_pytorch as smp
from fastai.vision.learner import create_body
import fastai.vision.models.xresnet as xresnet
import fastai.vision.models as models



class xResnetEncoder(torch.nn.Module, smp.encoders._base.EncoderMixin):

    def __init__(self, arch, out_channels, module_index, depth, **kwargs):
        super().__init__()
        self.backbone = create_body(models.xresnet.__dict__[arch], 3, pretrained=False, cut=None)

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels: List[int] = out_channels

        self._module_index: List[int] = module_index

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning fatures
        self._depth: int = depth

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3

        # Define enoder modules below
        ...

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """
        feature_list = []
        if -1 in self._module_index:
            feature_list.append(x)
        for i, module in enumerate(self.backbone._modules.values()):
            x = module(x)
            if i in self._module_index:
                feature_list.append(x)

        return feature_list

smp.encoders.encoders["xresnet18"] = {
    "encoder": xResnetEncoder,
    "params": {
        "arch": "xresnet18",
        "out_channels": [3, 32, 64, 128, 256, 512],
        "module_index": [-1, 0, 3, 5, 6, 7, 8],
        "depth": 5
    },
}

smp.encoders.encoders["xresnet34"] = {
    "encoder": xResnetEncoder,
    "params": {
        "arch": "xresnet34",
        "out_channels": [3, 32, 64, 128, 256, 512],
        "module_index": [-1, 0, 3, 5, 6, 7, 8],
        "depth": 5
    },
}

