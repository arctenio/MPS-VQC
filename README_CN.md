# MPS-VQC

这是 **MPS-VQC** 的官方代码仓库，对应一个面向**多模态人脸反欺诈（人脸活体检测）**的混合框架。

## 项目简介

当前仓库公开的是论文方法中的 **VQC 部分**，以及该模块使用的输入样例数据。

本项目研究任务为**人脸反欺诈 / 人脸活体检测**。论文中使用的原始公开基准数据集为 **CASIA-SURF**。当前仓库提供的 `.npz` 文件**不是原始 CASIA-SURF 数据**，而是基于原始多模态数据提取特征后，再经过 **MPS 模块压缩与融合**得到的特征数据，作为 **VQC 阶段的输入**。

## 当前仓库内容

当前公开版本包含：

- `mindquantum_circuit_acer_log.py`：对应论文中的 **VQC 部分实现**。
- `data_examples/20260104_232928_fused_features_simple_dim128_chi32.npz`
- `data_examples/20260104_233741_fused_features_activated_dim128_chi32.npz`

上述两个 `.npz` 文件为经过 **MPS 多模态融合与压缩**后得到的 VQC 输入样例数据。

## 与完整方法的关系

论文中的完整方法包含多个阶段。当前仓库主要公开 **VQC 阶段**及其对应输入样例；**单模态特征提取**部分和**基于 MPS 的多模态融合/压缩**部分代码将在后续逐步补充。

## MindSpore Quantum

本项目中的量子数值模拟基于 **MindSpore Quantum**。

```bibtex
@article{ma2023mindspore,
  title={MindSpore Quantum: A user-friendly, high-performance, and AI-compatible quantum computing framework},
  author={Ma, Hao and Liu, Yuxuan and Wang, Mingsheng and Zhao, Yilun and Wang, Yilin and Yang, Peng and Wang, Shengju and Xiong, Yao and He, Shaowei and Wang, Peina and others},
  journal={Software Impacts},
  volume={17},
  pages={100512},
  year={2023},
  publisher={Elsevier}
}
```

## 后续更新计划

后续将逐步补充：

- **单模态特征提取**代码；
- **基于 MPS 的多模态融合/压缩**代码；
- 更完整的**复现实验说明与项目文档**。

## License

本仓库计划采用 **Apache License 2.0**。
