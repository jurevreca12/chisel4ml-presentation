from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import BitWidthImplType
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import StatsOp
from brevitas.quant.solver import ActQuantSolver
from brevitas.quant.solver import BiasQuantSolver
from brevitas.quant.solver import WeightQuantSolver
from dependencies import value


class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


class WeightPerChannelQuant(WeightQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.PARAMETER_FROM_STATS
    )  # scale based on statistics
    scaling_stats_op = StatsOp.MAX  # scale statistics is the absmax value
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = True
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-127,127] rather than [-128, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.


class WeightPerTensorQuant(WeightQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.PARAMETER_FROM_STATS
    )  # scale based on statistics
    scaling_stats_op = StatsOp.MAX  # scale statistics is the absmax value
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = False
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-127,127] rather than [-128, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.


class IntBiasQuant(BiasQuantSolver):
    quant_type = QuantType.INT  # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = ScalingImplType.PARAMETER
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    requires_input_bit_width = False
    requires_input_scale = False
    scaling_per_output_channel = False
    bit_width = 8
    scaling_init = 2 ** (bit_width - 1)
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-127,127] rather than [-128, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.


class IntActQuant(ActQuantSolver):
    bit_width_impl_type = BitWidthImplType.CONST  # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND  # round to nearest
    scaling_impl_type = (
        ScalingImplType.CONST
    )  # scale is a parameter initialized from statistics
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    scaling_per_output_channel = False  # scale is per tensor
    signed = True  # quantization range is signed
    narrow_range = False  # quantization range is [-128, 127] rather than [-127, 127]
    zero_point_impl = ZeroZeroPoint  # zero point is 0.

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT