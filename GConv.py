import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp
import einops as ein


class GConv(hk.Module):
    def __init__(self, width, depth=96, bidirectional=True):
        self.width = width
        self.depth = depth
        self.bidirectional = bidirectional

        @hk.transparent
        def kernel(self, seq_length):
            scale_count = np.ceil(np.log(seq_length) / np.log(2)).astype(int)
            scales = 1 / 2 ** jnp.arange(scale_count)
            concat = []
            kernel = hk.get_parameter(
                "kernel", (self.width, self.depth), init=hki.RandomNormal()
            )
            for i, scale in enumerate(scales):
                concat.append(
                    jax.image.resize(
                        kernel * scale, (self.width * 2**i, self.depth), method="bilinear"
                    )
                )
            kernel = jnp.concat(concat)
            if self.bidirectional:
                kernel = ein.rearrange("(k n) d -> k n d", k=2)
                kernel = jnp.concatenate([kernel, kernel], axis=0)
            kernel = jnp.take(kernel, jnp.arange(seq_length), axis=0)
        return kernel

    def __call__(self, signal):
        seq_length = signal.shape[-2]
        k_f = jnp.fft.rfft(self.kernel(seq_length), axis=-2)
        u_f = jnp.fft.rfft(signal, axis=-2)
        y_f = k_f * u_f
        y = jnp.fft.irfft(y_f)
        b = hk.get_parameter("bias", self.depth)
        return y + b
