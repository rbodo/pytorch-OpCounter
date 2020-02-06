import torch
from rbodo.models import MlpIn, MlpOut, CnnIn, CnnOut, RnnOut
from thop import profile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()


input_shape = (100, 64)
num_frames = 10
num_embed = 128
num_sweeps = 20
num_classes = 5

print('MLP')
ops_mlp = []
params_mlp = []
neurons_mlp = []
for s in range(num_sweeps):
    _num_embed = num_embed * (s + 1) * 2
    shape_in = (num_frames, int(np.prod(input_shape)),)
    model_in = MlpIn(input_shape, _num_embed)
    model_out = MlpOut(_num_embed, num_frames, num_classes)
    macs_in, params_in, neurons_in, macs_per_layer_in, params_per_layer_in, neurons_per_layer_in = \
        profile(model_in, (torch.randn(shape_in),))

    shape_out = (1, _num_embed * num_frames)
    macs_out, params_out, neurons_out, macs_per_layer_out, params_per_layer_out, neurons_per_layer_out = \
        profile(model_out, (torch.randn(shape_out),))

    ops_mlp.append(macs_in + macs_out)
    params_mlp.append(params_in + params_out)
    neurons_mlp.append(neurons_in + neurons_out)

print('CNN')
ops_cnn = []
params_cnn = []
neurons_cnn = []
for s in range(num_sweeps):
    num_channels = 6 * (s + 1)
    shape_in = (num_frames, 1, *input_shape)
    model_in = CnnIn(num_embed, num_frames, num_channels)
    model_out = CnnOut(num_embed, num_classes)
    macs_in, params_in, neurons_in, macs_per_layer_in, params_per_layer_in, neurons_per_layer_in = \
        profile(model_in, (torch.randn(shape_in),))

    shape_out = (1, num_embed)
    macs_out, params_out, neurons_out, macs_per_layer_out, params_per_layer_out, neurons_per_layer_out = \
        profile(model_out, (torch.randn(shape_out),))

    ops_cnn.append(macs_in + macs_out)
    params_cnn.append(params_in + params_out)
    neurons_cnn.append(neurons_in + neurons_out)

print('RNN')
ops_rnn = []
params_rnn = []
neurons_rnn = []
for s in range(num_sweeps):
    shape_in = (num_frames, 1, *input_shape)
    model_in = CnnIn(num_embed, num_frames, 6)
    model_out = RnnOut(num_embed, num_classes, (s + 1) * 4)
    macs_in, params_in, neurons_in, macs_per_layer_in, params_per_layer_in, neurons_per_layer_in = \
        profile(model_in, (torch.randn(shape_in),))

    shape_out = (1, num_frames, 3 * 42)
    macs_out, params_out, neurons_out, macs_per_layer_out, params_per_layer_out, neurons_per_layer_out = \
        profile(model_out, (torch.randn(shape_out),))

    ops_rnn.append(macs_in + macs_out)
    params_rnn.append(params_in + params_out)
    neurons_rnn.append(neurons_in + neurons_out)

normalize = True
log = False

if normalize:
    neurons_mlp = np.divide(neurons_mlp, np.min(neurons_mlp))
    neurons_cnn = np.divide(neurons_cnn, np.min(neurons_cnn))
    neurons_rnn = np.divide(neurons_rnn, np.min(neurons_rnn))
    ops_mlp = np.divide(ops_mlp, np.min(ops_mlp))
    ops_cnn = np.divide(ops_cnn, np.min(ops_cnn))
    ops_rnn = np.divide(ops_rnn, np.min(ops_rnn))
    params_mlp = np.divide(params_mlp, np.min(params_mlp))
    params_cnn = np.divide(params_cnn, np.min(params_cnn))
    params_rnn = np.divide(params_rnn, np.min(params_rnn))

# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.scatter(neurons_mlp, ops_mlp, label='mlp')
plt.scatter(neurons_cnn, ops_cnn, label='cnn')
plt.scatter(neurons_rnn, ops_rnn, label='rnn')
if log:
    plt.xscale('log')
    plt.yscale('log')
if normalize:
    plt.xlabel('Scale factor neurons')
    plt.ylabel('Scale factor parameters')
else:
    plt.xlabel('# neurons')
    plt.ylabel('# ops')
plt.legend()
plt.savefig('ops')
plt.close()

plt.scatter(neurons_mlp, params_mlp, label='mlp')
plt.scatter(neurons_cnn, params_cnn, label='cnn')
plt.scatter(neurons_rnn, params_rnn, label='rnn')
if log:
    plt.xscale('log')
    plt.yscale('log')
if normalize:
    plt.xlabel('Scale factor neurons')
    plt.ylabel('Scale factor parameters')
else:
    plt.xlabel('# neurons')
    plt.ylabel('# params')
plt.legend()
plt.savefig('params')
plt.close()
