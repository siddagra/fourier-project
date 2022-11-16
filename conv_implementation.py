from typing import Iterable, Tuple, Union
from torch import nn
import torch.nn.functional as f


class FFTConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        ndim: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )

        kernel_size = to_ntuple(kernel_size, ndim)
        weight = torch.randn(out_channels, in_channels // groups, *kernel_size)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    def forward(self, signal):

        # Cast padding, stride & dilation to tuples.
        n = signal.ndim - 2
        padding_ = to_ntuple(padding, n=n)
        stride_ = to_ntuple(stride, n=n)
        dilation_ = to_ntuple(dilation, n=n)

        # internal dilation offsets
        offset = torch.zeros(
            1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
        offset[(slice(None), slice(None), *((0,) * n))] = 1.0

        # correct the kernel by cutting off unwanted dilation trailing zeros
        cutoff = tuple(slice(None, -d + 1 if d != 1 else None)
                       for d in dilation_)

        # pad the kernel internally according to the dilation parameters
        kernel = torch.kron(kernel, offset)[
            (slice(None), slice(None)) + cutoff]

        # Pad the input signal & kernel tensors
        signal_padding = [p for p in padding_[::-1] for _ in range(2)]
        signal = f.pad(signal, signal_padding, mode=padding_mode)

        # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
        # have *even* length.  Just pad with one more zero if the final dimension is odd.
        if signal.size(-1) % 2 != 0:
            signal_ = f.pad(signal, [0, 1])
        else:
            signal_ = signal

        kernel_padding = [
            pad
            for i in reversed(range(2, signal_.ndim))
            for pad in [0, signal_.size(i) - kernel.size(i)]
        ]
        padded_kernel = f.pad(kernel, kernel_padding)

        # Perform fourier convolution -- FFT, matrix multiply, then IFFT
        # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
        signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
        kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))

        kernel_fr.imag *= -1
        output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
        output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

        # Remove extra padded values
        crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
            slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
            for i in range(2, signal.ndim)
        ]
        output = output[crop_slices].contiguous()

        # Optionally, add a bias term before returning.
        if bias is not None:
            bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
            output += bias.view(bias_shape)

        return output
