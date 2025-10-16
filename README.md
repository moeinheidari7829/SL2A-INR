# SL²A-INR: Single-Layer Learnable Activation for Implicit Neural Representation

[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)](https://arxiv.org/abs/2409.10836)
[![arXiv](https://img.shields.io/badge/arXiv-2409.10836-b31b1b.svg)](https://arxiv.org/abs/2409.10836)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://YOUR-USERNAME.github.io/SL2A-INR/)

**[Project Website](https://YOUR-USERNAME.github.io/SL2A-INR/)** | **[Paper](https://arxiv.org/abs/2409.10836)** | **[Code](https://github.com/Iceage7/SL2A-INR)**

---

## Abstract

Implicit Neural Representation (INR), leveraging a neural network to transform coordinate input into corresponding attributes, has recently driven significant advances in several vision-related domains. However, the performance of INR is heavily influenced by the choice of the nonlinear activation function used in its multilayer perceptron (MLP) architecture.

We propose **SL²A-INR**, a hybrid network that combines a single-layer learnable activation function with an MLP that uses traditional ReLU activations. Our method achieves superior performance across diverse tasks, including image representation, 3D shape reconstruction, and novel view synthesis.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python train.py --input ./data/00.png --inr_model sl2a
```

## Citation

```bibtex
@article{heidari2024sl2a,
  title={SL$^{2}$A-INR: Single-Layer Learnable Activation for Implicit Neural Representation},
  author={Heidari, Moein and Rezaeian, Reza and Azad, Reza and 
          Merhof, Dorit and Soltanian-Zadeh, Hamid and Hacihaliloglu, Ilker},
  journal={arXiv preprint arXiv:2409.10836},
  year={2024},
  note={Accepted to ICCV 2025}
}
```

## Contact

For questions: [moein.heidari@ubc.ca](mailto:moein.heidari@ubc.ca)

## Acknowledgements

This work was supported by the Canadian Foundation for Innovation-John R. Evans Leaders Fund (CFI-JELF), Mitacs Accelerate program, and NSERC.
