from torch import Tensor
from typing import Union, Optional
from torch.nn.common_types import _size_2_t
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch import fft


def fft_conv(
    input: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    stride: _size_2_t = 1,
    padding: _size_2_t = 0,
    dilation: _size_2_t = 0,
    groups: int = 1,
) -> Tensor:
    # pad input
    padded_input = F.pad(input, (0,  # channel padding
                                 # height padding
                                 input.size(-1) - kernel.size(-1), input.size(-1) - \
                                 kernel.size(-1),
                                 input.size(-2) - kernel.size(-2), input.size(-2) - kernel.size(-2))  # width padding
                         )

    # make dimension even so fft is computed properly
    if input.size(-1) % 2 != 0:
        input = F.pad(input, [0, 1])

    # pad kernel
    padded_kernel = F.pad(kernel, (0,  # channel padding
                                   # height padding
                                   input.size(-1) - kernel.size(-1), input.size(-1) - \
                                   kernel.size(-1),
                                   input.size(-2) - kernel.size(-2), input.size(-2) - kernel.size(-2))  # width padding
                          )

    # fft on last two dimensions for depthwise seperable
    input_ft = fft.rfftn(padded_input, dim=(-1, -2))
    kernel_ft = fft.rfftn(padded_kernel, dim=(-1, -2))

    kernel_ft.imag *= -1  # take complex conjugate for cross-correlation

    output_ft = input_ft * kernel_ft  # convolution

    output = fft.irfftn(output_ft, dim=(-1, -2))  # inverse fft

    if bias is not None:
        output += bias.view(1, -1, 1)

    return output


class ConvFFT2D(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(ConvFFT2D, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return fft_conv(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return fft_conv(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
