# 音轨分离模型启动说明

## 训练环境：

- 本机：
  - CPU：Intel Core I9-14900HX
  - GPU：NVIDIA 4060 LAPTOP 8GB
  - RAM：32G
  - 硬盘：4T
- AutoDL：
  - GPU：NVIDIA 5090 32GB

## 数据集：

- MedleyDB（70.8GB）
- moisesdb_v0.1（146GB）

## 模型结构：

根据Demucs研究以及最新文献，先采用U-Net模型训练，后期为提升准确率可以往Reformor模型转换。连接层可以用transformer或者LSTM提升性能。

**使用SDR作为判断标准。**

## 参考文献

见article文件夹。