import brevitas.function.ops as bops
import brevitas.nn as qnn
import numpy as np
import torch
from torch.nn import Module
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from quantizers import CommonWeightQuant
from quantizers import IntBiasQuant
from quantizers import IntActQuant


def get_lhc_jets_model(bitwidth, use_bn=False):
    """Creates a model for LHC Jets dataset. Based on:
    https://github.com/fastmachinelearning/hls4ml-tutorial
    /blob/main/part1_getting_started.ipynb"""

    class LHCJetsModel(Module):
        "Four layer fully-connected model."
        def __init__(self, bw, use_bn):
            super().__init__()
            self.ishape = (1, 16)
            self.bw = bw
            self.use_bn = use_bn
            self.linear0 = qnn.QuantLinear(
                in_features=16,
                out_features=64,
                bias=True,
                weight_quant=CommonWeightQuant,
                weight_bit_width=bw,
                weight_scaling_impl_type="const",
                weight_scaling_init=1 if bw == 1 else 2 ** (bw - 1) - 1,
                bias_quant=IntBiasQuant,
                bias_bit_width=4,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (4 - 1) - 1,
                input_quant=IntActQuant,
                input_bit_width=8,
                input_scaling_impl_type="const",
                input_scaling_init=2 ** (8 - 1),
            )
            if self.use_bn:
                self.bn0 = torch.nn.BatchNorm1d(64)
            self.linear1 = qnn.QuantLinear(
                in_features=64,
                out_features=32,
                bias=True,
                weight_quant=CommonWeightQuant,
                weight_bit_width=bw,
                weight_scaling_impl_type="const",
                weight_scaling_init=1 if bw == 1 else 2 ** (bw - 1) - 1,
                bias_quant=IntBiasQuant,
                bias_bit_width=4,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (4 - 1) - 1,
            )
            if self.use_bn:
                self.bn1 = torch.nn.BatchNorm1d(32)
            self.linear2 = qnn.QuantLinear(
                in_features=32,
                out_features=32,
                bias=True,
                weight_quant=CommonWeightQuant,
                weight_bit_width=bw,
                weight_scaling_impl_type="const",
                weight_scaling_init=1 if bw == 1 else 2 ** (bw - 1) - 1,
                bias_quant=IntBiasQuant,
                bias_bit_width=4,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (4 - 1) - 1,
            )
            if self.use_bn:
                self.bn2 = torch.nn.BatchNorm1d(32)
            self.linear3 = qnn.QuantLinear(
                in_features=32,
                out_features=5,
                bias=True,
                weight_quant=CommonWeightQuant,
                weight_bit_width=bw,
                weight_scaling_impl_type="const",
                weight_scaling_init=1 if bw == 1 else 2 ** (bw - 1) - 1,
                bias_quant=IntBiasQuant,
                bias_bit_width=4,
                bias_scaling_impl_type="const",
                bias_scaling_init=2 ** (4 - 1) - 1,
            )
            if self.use_bn:
                self.bn3 = torch.nn.BatchNorm1d(5)
            self.quant_out = qnn.QuantIdentity(
                bit_width=4,
                scaling_impl_type="const",
                scaling_init=2 ** 3,
                signed=True
            )
            if bw > 1:
                self.act = qnn.QuantReLU(
                    bit_width=bw,
                    scaling_impl_type="const",
                    scaling_init=2 ** bw - 1
                )
            else:
                self.act = qnn.QuantIdentity(
                    bit_width=1,
                    scaling_impl_type="const",
                    scaling_init=1,
                    signed=True,
                )
        def forward(self, x):
            x = self.linear0(x)
            if self.use_bn:
                x = self.bn0(x)
            x = self.act(x)
            x = self.linear1(x)
            if self.use_bn:
                x = self.bn1(x)
            x = self.act(x)
            x = self.linear2(x)
            if self.use_bn:
                x = self.bn2(x)
            x = self.act(x)
            x = self.linear3(x)
            if self.use_bn:
                x = self.bn3(x)
            x = self.quant_out(x)
            return x

    model = LHCJetsModel(bitwidth, use_bn)
    return model

def get_lhc_jets_model_float():
    """Creates a model for LHC Jets dataset. Based on:
    https://github.com/fastmachinelearning/hls4ml-tutorial
    /blob/main/part1_getting_started.ipynb"""

    class LHCJetsModelFloat(Module):
        "Four layer fully-connected model."
        def __init__(self):
            super().__init__()
            self.ishape = (1, 16)
            self.linear0 = torch.nn.Linear(
                in_features=16,
                out_features=64,
                bias=True,
            )
            self.bn0 = torch.nn.BatchNorm1d(64)
            self.linear1 = torch.nn.Linear(
                in_features=64,
                out_features=32,
                bias=True,
            )
            self.bn1 = torch.nn.BatchNorm1d(32)
            self.linear2 = torch.nn.Linear(
                in_features=32,
                out_features=32,
                bias=True,
            )
            self.bn2 = torch.nn.BatchNorm1d(32)
            self.linear3 = qnn.QuantLinear(
                in_features=32,
                out_features=5,
                bias=True,
            )
            self.bn3 = torch.nn.BatchNorm1d(5)
            self.softmax = torch.nn.Softmax()

        def forward(self, x):
            x = self.linear0(x)
            x = self.bn0(x)
            x = torch.nn.ReLU(x)
            x = self.linear1(x)
            x = self.bn1(x)
            x = torch.nn.ReLU(x)
            x = self.linear2(x)
            x = self.bn2(x)
            x = torch.nn.ReLU(x)
            x = self.linear3(x)
            x = self.bn3(x)
            x = self.softmax(x)
            return x


    model = LHCJetsModelFloat()
    return model