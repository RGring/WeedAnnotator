#
# converts a saved PyTorch model to ONNX format
# 
import os
import argparse

import torch
import torchvision.models as models
import segmentation_models_pytorch as smp


# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='model_best.pth', help="path to input PyTorch model (default: model_best.pth)")
parser.add_argument('--output', type=str, default='', help="desired path of converted ONNX model (default: <ARCH>.onnx)")

opt = parser.parse_args() 
print(opt)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))

# load the model checkpoint
print('loading checkpoint:  ' + opt.input)
checkpoint = torch.load(opt.input)
print(checkpoint.keys())
print(checkpoint["config"])

# ToDo: Parametrize (save in checkpoint)
num_classes = checkpoint["config"]["num_classes"]
resolution = checkpoint["config"]["input_size"]
model = smp.__dict__[checkpoint["config"]["arch"]](
   encoder_name=checkpoint["config"]["encoder"],
   encoder_weights=None,
   classes = num_classes,
   activation=checkpoint["config"]["activation"],
)
if "efficientnet" in checkpoint["config"]["encoder"]:
   model.encoder.set_swish(memory_efficient=False)

# load the model weights
model.load_state_dict(checkpoint["state_dict"])

model.to(device)
model.eval()

print(model)
print('')

# create example image data
# 
input = torch.ones((1, 3, resolution[1], resolution[0])).cuda()
print('input size:  {:d}x{:d}'.format(resolution[1], resolution[0]))

# format output model path
if not opt.output:
   opt.output = f"{os.path.dirname(opt.input)}/model.onnx"

# export the modeli
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to ONNX...')
torch.onnx.export(model, input, opt.output, verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
print('model exported to:  {:s}'.format(opt.output))


