# README

## 解决方案

根据雷达、降水、风速等数据的特点，本项目测试滑动窗口预测下一帧图像与整体20帧图像预测。本项目提交的代码为整体预测。

### 滑动窗口

滑动窗口可以学习到图像中的动态特征。但是滑动窗口会导致误差的累计，导致生成图像的模糊。

### 整体预测

直接20帧图像的预测虽然预测的图像动态效果不好，但是会有一个较高的清晰度。

### 图像分辨率

由于数据大小为480×560,不是正方形，本项目通过补0的方式将数据调整为560×560,为了提高生成数据的清晰度。

## 模型构成

本项目采用生成对抗网络（GAN）进行预测，其中生成器采用双层ConvLSTM进行时空特征学习。UNet进行20帧图像的预测。判别器采用单层ConvLSTM进行时空特征提取，通过多层二维卷积进行特征聚合，最后对输入图像进行判别。

## 使用

### 环境与依赖

- 操作系统
  - Linux version 4.15.0-177-generic (buildd@ubuntu) 
  - (gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04)) 
  - #186-Ubuntu SMP Thu Apr 14 20:23:07 UTC 2022

- CUDA 10.2

- Python 3.6.13

- cudnn :  7.6

- pytorch 1.9.0

- Visdom

- cv2等

  ### 运行

  可以在./code/config.py中修改训练与测试的超参数。比如epoch，batchsize，use_gpu等等，值得注意的是由于设备有限，本次只训练了训练集的千分之一，并且训练了10个epoch，相信更好的设备参考滑动窗的预测方式会有更好的效果，代码中可以修改成滑动窗训练，但是对应匹配的main.py文件并没有提交，可以通过修改提交的main.py文件获得。

  ### 温馨提示

  不同数据的训练需要修改./code/config.py文件中datatype、load_model_path和load_G_model_path。
  
  #### 训练数据

  - 比如测试radar数据训练，需要将./code/config.py文件中datatype、load_model_path和load_G_model_path改为
  
  - ```shell
    python main.py train --load_model_path "" --load_G_model_path "" --datatype 'radar'
    ```

    同理 wind数据与precip数据分别为：

    ```shell
    python main.py train --load_model_path "" --load_G_model_path "" --datatype 'wind'
    ```
  
    ```shell
    python main.py train --load_model_path "" --load_G_model_path "" --datatype 'precip'
    ```
  
  但你不需要修改factor，因为这个参数并没有用到。
  
  #### 训练数据（预训练版本）
  
  - 比如测试radar数据训练，需要将./code/config.py文件中datatype、load_model_path和load_G_model_path改为
  
  - ```shell
    python main.py train --load_model_path "../user_data/Dwind.pth" --load_G_model_path "../user_data/Gwind.pth" --datatype 'radar'
    ```
    
    同理wind数据与precip数据分别为：
  
    ```shell
    python main.py train --load_model_path "../user_data/Dwind.pth" --load_G_model_path "../user_data/Gwind.pth" --datatype 'wind'
    ```
    
    ```shell
    python main.py train --load_model_path "../user_data/Dwind.pth" --load_G_model_path "../user_data/Gwind.pth" --datatype 'precip'
    ```
  
  但你不需要修改factor，因为这个参数并没有用到。
  
  #### 测试数据
  
  如果你只是想通过训练好的权重获得B榜预测结果只需要运行在code路径下执行以下命令。
  
  ```shell
  python main.py test 
  ```
  
  
  
  预测A榜的结果需要修改
  
  ``` python
   testpath = r'..data//TestB/{}/'
  ```
  
  