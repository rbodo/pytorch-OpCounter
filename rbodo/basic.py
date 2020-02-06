import torch
from rbodo.models import MlpIn, MlpOut, CnnIn, CnnOut, RnnOut
from rbodo.models2 import SimpleConv2d
from thop import profile
import numpy as np

input_shape = (100, 64)
num_frames = 10
num_embed = 128

models = {'MLP': (MlpIn(), MlpOut(), (num_frames, int(np.prod(input_shape)),), (1, num_embed * num_frames)),
          'CNN': (CnnIn(), CnnOut(), (num_frames, 1, *input_shape), (1, num_embed)),
          'RNN': (CnnIn(), RnnOut(), (num_frames, 1, *input_shape), (1, num_frames, 3 * 42))}

for name, (model_in, model_out, shape_in, shape_out) in models.items():
    print(name)
    macs_in, params_in, neurons_in, macs_per_layer_in, params_per_layer_in, neurons_per_layer_in = \
        profile(model_in, (torch.randn(shape_in),))

    macs_out, params_out, neurons_out, macs_per_layer_out, params_per_layer_out, neurons_per_layer_out = \
        profile(model_out, (torch.randn(shape_out),))

    total_macs_per_layer = {}
    total_params_per_layer = {}
    total_neurons_per_layer = {}
    total_macs_per_layer.update(macs_per_layer_in)
    total_macs_per_layer.update(macs_per_layer_out)
    total_params_per_layer.update(params_per_layer_in)
    total_params_per_layer.update(params_per_layer_out)
    total_neurons_per_layer.update(neurons_per_layer_in)
    total_neurons_per_layer.update(neurons_per_layer_out)

    print(total_macs_per_layer)
    print(total_params_per_layer)
    print(total_neurons_per_layer)
    print("Total macs: {}".format(macs_in + macs_out))
    print("Total params: {}".format(params_in + params_out))
    print("Total neurons: {}".format(neurons_in + neurons_out))

print('SimpleConv2d')
model = SimpleConv2d(1, 8, 3)
shape = (1, 1, 32, 32)
macs, params, neurons, macs_per_layer, params_per_layer, neurons_per_layer = \
    profile(model, (torch.randn(shape),))

print(macs_per_layer)
print(params_per_layer)
print(neurons_per_layer)
print("Total macs: {}".format(macs))
print("Total params: {}".format(params))
print("Total neurons: {}".format(neurons))
