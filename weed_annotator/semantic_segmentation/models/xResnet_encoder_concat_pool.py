import torch
from typing import List
import segmentation_models_pytorch as smp
from fastai.vision.learner import create_body
import fastai.vision.models.xresnet as xresnet
import fastai.vision.models as models
from fastai.vision.all import AdaptiveConcatPool2d




class xResnetEncoder(torch.nn.Module, smp.encoders._base.EncoderMixin):

    def __init__(self, arch, out_channels, module_index, depth, concat_pool, input_size, **kwargs):
        super().__init__()
        self.do_pool = concat_pool
        self.backbone = create_body(models.xresnet.__dict__[arch], 3, pretrained=False, cut=None)
        self._depth: int = len(out_channels) - 1
        if self.do_pool:
            out_size = (int(input_size[0]/(2**self._depth)), int(input_size[1]/(2**self._depth)))
            self.backbone._modules[str(module_index[-1])] = AdaptiveConcatPool2d(size=out_size)

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels: List[int] = out_channels
        # if self.do_pool:
        #     self._out_channels.append(1024)

        self._module_index: List[int] = module_index

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning fatures
        # if self.do_pool:
        #     self._depth += 8

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3


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

smp.encoders.encoders["xcpresnet18"] = {
    "encoder": xResnetEncoder,
    "params": {
        "arch": "xresnet18",
        "out_channels": [3, 64, 64, 128, 256, 512, 1024],
        "module_index": [-1, 2, 4, 5, 6, 7, 8],
        "concat_pool": True
    },
}

smp.encoders.encoders["xresnet18"] = {
    "encoder": xResnetEncoder,
    "params": {
        "arch": "xresnet18",
        "out_channels": [3, 64, 64, 128, 256, 512],
        "module_index": [-1, 2, 4, 5, 6, 7],
        "concat_pool": False
    },
}

smp.encoders.encoders["xcpresnet34"] = {
    "encoder": xResnetEncoder,
    "params": {
        "arch": "xresnet34",
        "out_channels": [3, 64, 64, 128, 256, 512, 1024],
        "module_index": [-1, 2, 4, 5, 6, 7, 8],
        "concat_pool": True
    },
}

smp.encoders.encoders["xresnet34"] = {
    "encoder": xResnetEncoder,
    "params": {
        "arch": "xresnet34",
        "out_channels": [3, 64, 64, 128, 256, 512],
        "module_index": [-1, 2, 4, 5, 6, 7],
        "concat_pool": False
    },
}

smp.encoders.encoders["xcpresnet50"] = {
    "encoder": xResnetEncoder,
    "params": {
        "arch": "xresnet50",
        "out_channels": [3, 64, 256, 512, 1024, 2048, 4096],
        "module_index": [-1, 2, 4, 5, 6, 7, 8],
        "concat_pool": True
    },
}

smp.encoders.encoders["xresnet50"] = {
    "encoder": xResnetEncoder,
    "params": {
        "arch": "xresnet50",
        "out_channels": [3, 64, 256, 512, 1024, 2048],
        "module_index": [-1, 2, 4, 5, 6, 7],
        "concat_pool": False
    },
}

