# MPS-VQC

Official code repository for MPS-VQC, a hybrid quantum-classical framework for multimodal face anti-spoofing (face liveness detection).

## Overview

This repository currently releases the VQC part of the method described in our paper, together with example input data used by the VQC module.

The task studied in this project is face anti-spoofing / face liveness detection. The original public benchmark used in the paper is CASIA-SURF. The `.npz` files provided in this repository are not raw CASIA-SURF data; they are features extracted from the original multimodal data and further compressed/fused by the MPS module, prepared as inputs for the VQC stage.

## Repository Status

The current public version of this repository includes:

- `mindquantum_circuit_acer_log.py`: implementation corresponding to the **VQC part** in the paper.
- `data_examples/20260104_232928_fused_features_simple_dim128_chi32.npz`
- `data_examples/20260104_233741_fused_features_activated_dim128_chi32.npz`

The two `.npz` files are example inputs for the VQC module after **MPS-based multimodal fusion and compression**.

## Relation to the Full Pipeline

The complete method in the paper contains multiple stages. At present, this repository mainly releases the **VQC stage** and its corresponding example inputs. The **single-modal feature extraction** part and the **MPS-based multimodal fusion/compression** part will be added later.

## MindSpore Quantum

The quantum simulation in this project is based on **MindSpore Quantum**.

```bibtex
@misc{xu2024mindspore, 
  title={MindSpore Quantum: A User-Friendly, High-Performance, and AI-Compatible Quantum Computing Framework}, 
  author={Xusheng Xu and Jiangyu Cui and Zidong Cui and Runhong He and Qingyu Li and Xiaowei Li and Yanling Lin and Jiale Liu and Wuxin Liu and Jiale Lu and others}, 
  year={2024}, 
  eprint={2406.17248}, 
  archivePrefix={arXiv}, 
  primaryClass={quant-ph}, 
  url={https://arxiv.org/abs/2406.17248}, 
}
```

## Planned Updates

The repository will be gradually expanded with:

- code for single-modal feature extraction;
- code for MPS-based multimodal fusion/compression;
- more complete reproduction instructions and project documentation.

## License

 **Apache License 2.0**.
