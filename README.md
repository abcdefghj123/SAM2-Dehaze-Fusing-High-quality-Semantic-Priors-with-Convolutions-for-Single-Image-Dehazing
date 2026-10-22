<img width="1411" height="855" alt="image" src="https://github.com/user-attachments/assets/76a13966-9c87-4bb9-ac03-d663cb8b968a" />[README.md](https://github.com/user-attachments/files/33819824/README.md)
<div align="center">

# SAM2-Dehaze

### Fusing High-Quality Semantic Priors with Convolutions for Single-Image Dehazing

**Sen Li · Jianchao Wang · Zhanqiang Huo**

*Sensors*, 2025, **25**(22), 7097

[![Paper](https://img.shields.io/badge/Paper-Sensors%202025-blue)](https://doi.org/10.3390/s25227097)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs25227097-green)](https://doi.org/10.3390/s25227097)

</div>
![Uploading image.png…]()

## Introduction

**SAM2-Dehaze** is a semantic-guided framework for **single-image dehazing** that integrates high-quality semantic priors extracted by **Segment Anything Model 2 (SAM2)** with convolutional feature learning. It is designed to address information loss and the insufficient exploitation of structural cues in hazy scenes, improving the reconstruction of textures, edges, and object boundaries.

Built on a five-stage U-Net architecture, SAM2-Dehaze injects SAM2-derived semantic maps into multiple restoration stages and combines semantic fusion, detail-enhanced convolution, and semantic alignment to generate visually natural dehazed images.

## Overall Architecture

<div align="center">
  <img src="assets/SAM2_Dehaze_Architecture.png" alt="Overall architecture of SAM2-Dehaze" width="100%">
  <p><em>Figure 1. Overall architecture of SAM2-Dehaze. SAM2-derived semantic priors guide the five-stage dehazing U-Net, and the network is optimized with L1 and contrastive losses.</em></p>
</div>

As shown above, a hazy image is processed by SAM2 to produce a semantic prior map. The map is fused into the restoration backbone at multiple scales. The network then reconstructs the dehazed image through convolutional feature processing, selective-kernel fusion, and a final semantic alignment stage.

## Key Components

- **Semantic Prior Fusion Block (SPFB):** Incorporates SAM2 semantic maps at different stages to guide structure-aware feature learning and improve the restoration of semantic boundaries and fine details.
- **Parallel Detail-Enhanced and Compression Convolution (PDCC):** Combines standard convolution, detail-enhancement/difference convolution, and spatial-channel reconstruction to model complementary features and recover high-frequency information.
- **Semantic Alignment Block (SAB):** Refines the reconstructed image with semantic guidance to maintain color, texture, and structural consistency.
- **Loss Function:** Uses a combination of **L1 reconstruction loss** and **contrastive loss** during training.

## Project Structure

The main training and testing entry points are:

```text
SAM2-Dehaze/
├── README.md
├── train.py                 # Training script
├── test.py                  # Testing / evaluation script
├── assets/
│   └── SAM2_Dehaze_Architecture.png
└── ...                      # Other project modules and configurations
```

> The structure above lists the confirmed entry-point scripts and the README image. Additional modules, datasets, and configuration files depend on the actual released code.

## Environment Setup

The experiments reported in the paper use **PyTorch 2.4.1** and an **NVIDIA RTX A6000 (48 GB)** GPU. The project also relies on semantic priors provided by **SAM2**.

1. Create and activate a Python environment suitable for the uploaded source code.
2. Install the dependencies required by the implementation. If the repository contains a `requirements.txt` file, you can install it with:

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the official [SAM2](https://github.com/facebookresearch/sam2) installation and pretrained checkpoint **if your implementation generates semantic priors at runtime**. Otherwise, prepare the semantic prior maps expected by your dataset loader.
4. Update the dataset paths, checkpoint paths, and configuration options in the released code as needed.

> **Note:** The Python version and complete dependency list should be taken from the actual code environment; they are not inferred from the paper alone.

## Training

The **`train.py`** file is the training entry point:

```bash
python train.py
```

Before running the command, configure the training dataset paths, SAM2 prior inputs (or SAM2 checkpoint), output checkpoint directory, and any required command-line arguments in `train.py`. The command above assumes the script can run with its configured defaults; check its argument parser if additional flags are required.

The published experiments use the **AdamW** optimizer, an initial learning rate of **2 × 10⁻⁴**, a cosine annealing schedule, and **256 × 256** image crops. Please refer to the paper for the full training protocol and dataset-specific settings.

## Testing

The **`test.py`** file is the testing/evaluation entry point:

```bash
python test.py
```

Before testing, set the pretrained dehazing model checkpoint, test dataset or input-image path, semantic prior inputs (if required), and output directory in `test.py`. Review the script for any required arguments.

For paired benchmark datasets, common evaluation metrics include **PSNR** and **SSIM**. Please refer to the paper for additional metrics and experimental details.

## Datasets

The published study evaluates SAM2-Dehaze on the following benchmarks:

| Dataset | Usage |
| :-- | :-- |
| **RESIDE ITS** | Synthetic indoor training |
| **RESIDE OTS** | Synthetic outdoor training |
| **SOTS-Indoor / SOTS-Outdoor** | Synthetic benchmark testing |
| **Dense-Haze** | Dense-haze evaluation |
| **NH-Haze** | Non-homogeneous-haze evaluation |
| **RTTS** | Real-world hazy-image evaluation |

Please obtain these datasets from their official distribution channels and comply with their respective usage terms.

## Quantitative Results

The following **SAM2-Dehaze-L** results are reported in the published paper:

| Dataset | PSNR (dB) ↑ | SSIM ↑ |
| :-- | --: | --: |
| **SOTS-Indoor** | **42.83** | **0.997** |
| **SOTS-Outdoor** | **36.22** | **0.989** |
| **Dense-Haze** | **20.61** | **0.725** |
| **NH-Haze** | **22.02** | **0.831** |

These are **paper-reported results**, not claims that the repository has already been independently reproduced. For full comparisons, model variants, efficiency analysis, qualitative examples, and ablation studies, see the [published article](https://doi.org/10.3390/s25227097).

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2025sam2dehaze,
  title   = {SAM2-Dehaze: Fusing High-Quality Semantic Priors with Convolutions for Single-Image Dehazing},
  author  = {Li, Sen and Wang, Jianchao and Huo, Zhanqiang},
  journal = {Sensors},
  volume  = {25},
  number  = {22},
  pages   = {7097},
  year    = {2025},
  doi     = {10.3390/s25227097}
}
```

## Acknowledgements

This project makes use of semantic priors derived from [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2). We thank the authors of SAM2 and the maintainers of the dehazing benchmarks used in this research.

## License

A repository license has not yet been specified. Before reusing or redistributing the code and third-party pretrained weights, check the applicable license terms and add a `LICENSE` file if appropriate.
