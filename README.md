# **小样本 皮肤病变 医学图像分割 一站式工具包 / FSL-SkinSeg-Toolkit**

- 本项目是基于我的毕业设计代码修改完善与整理得来，可能还有小问题以及部分完善空间，各位使用者遇到问题也可以多提出反馈
- 本项目是基于PH2皮肤病变数据进行训练的，因此读取数据集的部分是根据 **PH2数据集** 写的，需要读取其他数据集可以修改配置
- 因为本人电脑由于一些原因无法使用GPU训练而只能 **基于CPU** 进行训练，因此本项目在某些算法方面基于CPU进行了适配

- This is my graduation project.
- And this project trains model based on CPU, using PH2 dataset for training.

提示：以下提到的路径均从本项目的根目录（**FSL-SkinImgSeg-Toolkit**）开始.
tips：The following path are all from the root（**FSL-SkinImgSeg-Toolkit**）.  

# 使用流程 / Guide

## 预处理 / Preprocess

- datasets_splitor.py

  - **功能**：将原始数据集分割按6 : 2 : 2的比例分割成训练集、验证集和测试集

  - **使用方法**：

    - 以PH2数据集为例，先将解压好的各组文件按照原图与掩码的方式同一文件夹的方式，放到\images里面, 结构示范如下：
      - images
        - IMD003
          - IMD003_Dermoscopic_Image
          - IMD003_lesion
          - IMD003_roi
        - IMD004
          - 同上

    - 放好images文件之后运行 **datasets_splitor.py** 即可
    - **温馨提示**：运行程序前，若没有images和images_split文件夹，均无需手动创建，程序会自动创建

- image_to_pt.py

  - **功能**：用于将图像转换成能加速读取的，pytorch的pt文件，用空间换时间

  - **使用方法**：在Python 的程序主入口（就是`if __name__ == '__main__':`内）选择对应参数，运行 **image_to_pt.py** 即可



## 训练 / Train

- my_dataset.py / my_model.py / losses.py

  - **功能**：分别是训练所需的数据集的类、基础模型U-net以及训练模型用到的损失函数（包含自创的tversky_hd）  

- train.py

  - **功能**：训练模型用到的主要程序

  - **使用方法**：

    - 先到 **待训练任务清单 logs\tasks.yml** 配置需要训练的任务，之后点击运行 **Run_train.bat** （前提是已经激活python环境）或者直接运行 **train.py** 即可开启训练  
    - 在程序读取 **logs\tasks.yml** 内的任务后会删除，并把所有的训练历史记录到 **logs\tasks_history.yml** 中
    - **温馨提示**：首次运行程序前没有tasks.yml，直接运行本程序即可自动生成tasks.yml文件，可以根据里面预设的范例配置即可

- Run_train.bat（仅限window系统）

  - **功能**：不用打开编译器或者IDE，通过系统自带的命令行启动 **train.py** ，进行训练以及日志展示

  - **使用方法**：

    - 在已经激活python环境后，点击运行 **Run_train.bat** 即可
    - 本脚本有训练结束 **自动关机** 功能，通过一个命名为 **no_shutdown** 的文件的 **位置** 控制是否打开此功能
    - no_shutdown放在 **项目根目录** 下（即与Run_train.bat同一目录下）时，自动关机功能 **开启** ；no_shutdown放在其他位置时，自动关机功能 **关闭**



## 性能评估 / Evaluation

- logs_copier.py 

  - **功能**：批量迁移训练日志

  - **使用方法**：在Python的程序主入口配置好训练结果和输出的路径（均有预设默认值），运行代码 **logs_copier.py** 即可

- draw_TV_curve.py 

  - **功能**：绘制Train Loss和Val Loss随着epoch变化的曲线

  - **使用方法**：在Python的程序主入口填写好相关配置（颜色、绘图样式、读取和保存地址等配置均有预设默认值），运行代码 **logs_copier.py** 即可

- image_comparion.py 

  - **功能**：绘制不同模型对同一图像的预测结果的对照图（定性分析）

  - **使用方法**：在Python的程序主入口填写好相关配置（图组、所绘制用的图片ID等配置均有预设默认值），运行代码 **image_comparion.py** 即可

- metrics_evaluate.py 

  - **功能**：计算不同模型对图像识别的各项指标并生成csv表格文件（定量分析）

  - **使用方法**：在Python的程序主入口填写好相关配置（所需指标、模式切换、读取和保存地址等配置均有预设默认值），运行代码 **metrics_evaluate.py** 即可
  
- metrics_plot.py 

  - **功能**：根据生成的各项指标csv表格文件，生成可视化的柱状图（定量分析）

  - **使用方法**：在Python的程序主入口填写好相关配置（模式切换、读取和保存地址等配置均有预设默认值），运行代码 **metrics_plot.py** 即可