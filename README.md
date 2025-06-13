

# Scaling Up Computer Vision Neural Networks Using Fast Fourier Transform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Efficient neural network architectures for computer vision using Fast Fourier Transform to overcome quadratic complexity limitations in Vision Transformers and enable large kernel convolutions.

## 🚀 Overview

This project explores approaches to scale up neural networks for computer vision using Fast Fourier Transform (FFT):

1. **Fourier Image Transformers (FiT)** - Replace quadratic self-attention with O(n log n) FFT operations
2. **FFT-based Convolutions** - Enable arbitrarily large convolutional kernels with frequency domain operations

## 📊 Key Results

### Fourier Image Transformers vs Vision Transformers

| Model   | Accuracy  | Inference Time | Parameters |
| ------- | --------- | -------------- | ---------- |
| ViT     | 93.5%     | 5.67ms         | 86M        |
| **FiT** | **94.3%** | **3.6ms**      | **38M**    |

*CIFAR-10 benchmark results*

### FFT Convolutions Performance

| Model                 | Accuracy  | Inference Time | Parameters |
| --------------------- | --------- | -------------- | ---------- |
| RepLKNet-base         | 83.5%     | 41.2ms         | 79M        |
| **FFT-Conv-RepLKNet** | **83.4%** | **28.7ms**     | **79M**    |

*ImageNet-1k benchmark results*

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/siddagra/fourier-project.git
cd fourier-project

# Install dependencies
pip install torch torchvision
pip install numpy matplotlib
pip install jax jaxlib  # Optional, for advanced experiments
```

## 🏗️ Architecture Components

### 1. Fourier Image Transformers (FiT)

Replace quadratic self-attention in Vision Transformers with FFT operations:

```python
class FFTLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        return torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
```

**Key Benefits:**

* O(n log n) complexity vs O(n²) in standard ViT
* 56% fewer parameters (38M vs 86M)
* 36% faster inference (3.6ms vs 5.67ms)
* Better accuracy on CIFAR-10 (94.3% vs 93.5%)

### 2. FFT-based Convolutions

Enable large kernel convolutions using frequency domain operations:

```python
# Pad inputs for linear convolution
padded_image = F.pad(image, (kernel_size-1,), value=0.0)
padded_kernel = F.pad(kernel, image_size + kernel_size - 1, value=0.0)

# FFT-based convolution
image_ft = torch.fft.rfftn(padded_image, dim=(-1,-2))
kernel_ft = torch.fft.rfftn(padded_kernel, dim=(-1,-2))
kernel_ft.imag *= -1  # Cross-correlation instead of convolution
output_ft = image_ft * kernel_ft
output = torch.fft.irfftn(output_ft, dim=(-2,-1))
```

**Key Benefits:**

* O(n² log n) complexity independent of kernel size
* Direct replacement for existing CNN architectures
* 30% faster inference on RepLKNet
* Maintains accuracy with arbitrarily large kernels

## 📁 Project Structure

```
fourier-project/
├── FFTConv2D.py       # FFT-based convolution implementation
├── FiT.py             # Fourier Image Transformer implementation
├── GConv.py           # Additional convolution modules
├── config.json        # Configuration files
├── engine.py          # Training and evaluation engine
├── testfps.py         # Speed and performance tests
├── testfps-2.py
├── testfps-3.py
├── train.py           # Training script
├── transforms.py      # Data augmentation and preprocessing
└── LICENSE            # License file
```

## 🚀 Quick Start

### Training Fourier Image Transformer

```python
from FiT import FourierImageTransformer
from engine import train

config = {
    'img_size': (32, 32),
    'patch_size': (2, 2),
    'embed_dim': 256,
    'num_classes': 10
}

model = FourierImageTransformer(config)

train(model, dataset='cifar10', epochs=100)
```

### Using FFT Convolutions

```python
from FFTConv2D import FFTConv2d

conv = FFTConv2d(in_channels, out_channels, kernel_size=31)
output = conv(input_tensor)
```

## 📈 Scalability Analysis

FFT-based approaches demonstrate superior scaling:

| Image Size | Patch Size | FiT Time | ViT Time |
| ---------- | ---------- | -------- | -------- |
| 32×32      | 4×4        | 3.6ms    | 5.6ms    |
| 224×224    | 4×4        | 26.6ms   | 106.8ms  |
| 1080×1080  | 16×16      | 40.9ms   | DNF\*    |

\*DNF = Did Not Finish (out of memory)

## 🧪 Experiments

### CIFAR-10 Classification

```bash
python testfps.py --model fit --patch_size 2
```

### ImageNet Classification

```bash
python testfps-2.py --model fft_replknet --kernel_size 31
```

### Speed Benchmarking

```bash
python testfps-3.py --compare_all
```

## 🔬 Technical Details

* Uses PyTorch CUDA FFT for GPU acceleration
* Applies proper padding to convert circular convolution to linear convolution
* Extracts real part for numerical stability during FFT operations
* Automatic mixed precision disabled for FFT layers to ensure stability

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{agrawal2023scaling,
  title={Scaling Up Computer Vision Neural Networks Using Fast Fourier Transform},
  author={Agrawal, Siddharth},
  journal={arXiv preprint arXiv:2302.12185},
  year={2023}
}
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

* Vision Transformer (ViT) by Dosovitskiy et al.
* FNet architecture by Lee-Thorp et al.
* RepLKNet architecture by Ding et al.

---

⭐ **Star this repository if you found it helpful!**

---
