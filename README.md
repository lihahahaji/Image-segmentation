# Image-segmentation
人工智能物联网大作业

## 项目结构说明
```
Image-segmentation
├─ .gitignore
├─ doc_src -- 实验报告latex源码
├─ logs -- 模型训练日志
│  ├─ AttU_Net_train_6_14.log
│  ├─ FCN_train.log
│  └─ UNet_train_6_14.log
├─ README.md
├─ requirements.txt
├─ src
│  ├─ eval.py -- 模型测试
│  ├─ models -- 模型源码
│  │  ├─ att_unet.py
│  │  ├─ fcn.py
│  │  └─ u_net.py
│  ├─ test.py
│  └─ train -- 模型训练代码
│     ├─ train_attunet.py
│     ├─ train_fcn.py
│     └─ train_unet.py
└─ train_result_visualization -- 实验结果可视化
   ├─ AttU_Net
   ├─ FCN
   └─ UNet
```


## 数据集说明
Synapse数据集

### 目录结构

```
Synapse/
  ├─ lists/
  │  └─ lists_Synapse/
  │     ├─ all.lst
  │     ├─ test_vol.txt
  │     └─ train.txt
  ├─ test_vol_h5/
  │  ├─ case0001.npy.h5
  │  ├─ case0002.npy.h5
  │  └─ ...
  └─ train_npz/
     ├─ case0005_slice000.npz
     ├─ case0005_slice001.npz
     └─ ...
```

### 详细说明

#### 1. `lists/`
这个目录包含一些列表文件，这些文件列出了数据集中用于训练和测试的文件名。

- `all.lst`：包含数据集中所有文件的列表。通常用于参考或完整数据集的处理。
- `test_vol.txt`：包含用于测试的文件名列表。这些文件通常位于`test_vol_h5`目录中。
- `train.txt`：包含用于训练的文件名列表。这些文件通常位于`train_npz`目录中。

#### 2. `test_vol_h5/`
这个目录包含用于测试的样本，以`.h5`（HDF5）格式存储。每个文件名的格式为`caseXXXX.npy.h5`，其中`XXXX`是样本编号。

- `case0001.npy.h5`，`case0002.npy.h5`，...：这些文件包含测试样本的图像数据和对应的标签数据，存储在HDF5格式的文件中。每个文件通常包含一个三维的医学图像和相应的分割标签。

#### 3. `train_npz/`
这个目录包含用于训练的样本，以`.npz`格式存储。每个文件名的格式为`caseXXXX_sliceYYY.npz`，其中`XXXX`是样本编号，`YYY`是切片编号。

- `case0005_slice000.npz`，`case0005_slice001.npz`，...：这些文件包含训练样本的图像切片和对应的标签数据，存储在NumPy的`.npz`文件中。每个文件通常包含一个二维的医学图像切片和相应的分割标签。

### 文件内容

#### `.npz` 文件
这些文件是NumPy的压缩格式，可以存储多个数组。通常包含两个数组：
- `image`：一个二维数组，表示医学图像的某个切片。
- `label`：一个二维数组，表示对应的分割标签。

#### `.h5` 文件
这些文件是HDF5格式，可以存储多种数据类型，适合存储大型、多维数据。通常包含两个数据集：
- `image`：一个三维数组，表示整个医学图像。
- `label`：一个三维数组，表示对应的分割标签。


## 实验结果
### FCN

![image-20240616175051319](./train_result_visualization\FCN\FCNtraining_metrics_epochs_100.png)

### U-Net
![image-20240616175051319](./train_result_visualization\UNet\UNettraining_metrics_epochs_100.png)

### Attention U-Net

![image-20240616175051319](./train_result_visualization\AttU_Net\AttU_Net_training_metrics_epochs_100.png)