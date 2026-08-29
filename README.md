# **小样本皮肤病变医学图像分割一站式工具包 / FSL-SkinSeg-Toolkit**

- 本项目基于我的毕业设计代码修改、完善与整理而成。作为一个持续迭代的项目，可能仍存在些许问题与完善空间，欢迎各位使用者在遇到问题时积极提出反馈。
- 本项目默认基于 **PH2 皮肤病变数据集** 进行训练，因此数据读取模块专为 PH2 的目录结构设计。如需使用其他数据集，请自行修改相关配置。
- 受限于硬件条件，本项目最初为了纯 CPU 训练环境而生，因此在诸多底层算法方面针对 **纯 CPU 计算** 进行了深度优化与适配。

**温馨提示**：以下提到的路径均从本项目的根目录（**FSL-SkinImgSeg-Toolkit**）开始计算。

# 使用流程 / Guide

## 预处理 / Preprocess

- **`datasets_split.py`**
  - **功能**：将原始数据集按 6 : 2 : 2 的比例随机划分为训练集、验证集和测试集。
  - **使用方法**：
    - 以 PH2 数据集为例，先将解压好的各组文件（原图与掩码处于同一文件夹）放入 `\images` 目录下，结构示范如下：
      - `images`
        - `IMD003`
          - `IMD003_Dermoscopic_Image`
          - `IMD003_lesion`
          - `IMD003_roi`
        - `IMD004` (同上)
    - 放好 `images` 文件夹后，直接运行 **`datasets_split.py`** 即可。
    - **注**：运行程序前无需手动创建 `images` 和 `images_split` 文件夹，程序执行时会自动生成。

- **`image_to_pt.py`**
  - **功能**：将图像数据转换为能够加速读取的 PyTorch `.pt` 格式文件，用空间换取训练时间。
  - **使用方法**：在 Python 的程序主入口（即 `if __name__ == '__main__':` 内部）选择对应参数，随后运行 **`image_to_pt.py`** 即可。

## 训练 / Train

- **`my_dataset.py` / `my_model.py` / `losses.py`**
  - **功能**：基础核心组件模块，分别对应训练所需的数据集类、基础 U-Net 模型架构，以及训练时使用的损失函数库（包含自创的 `TverskyHausdorffLoss`）。

- **`train.py`**
  - **功能**：执行模型训练的核心主程序。
  - **使用方法**：
    - 先前往待训练任务清单 **`logs\tasks.yml`** 配置你需要运行的实验参数。
    - 配置完成后，双击运行 **`Run_train.bat`**（前提是已配置好 Python 环境变量），或在终端直接运行 **`train.py`** 开启自动化训练。
    - 程序读取 **`logs\tasks.yml`** 内的任务后会将其弹出队列，并将所有的训练历史与时间戳归档至 **`logs\tasks_history.yml`** 中。
    - **注**：首次运行程序前如果没有 `tasks.yml`，直接运行一次本程序即可自动生成模板文件，后续只需根据里面的预设范例修改即可。

- **`Run_train.bat`** (仅限 Windows 系统)
  - **功能**：无需打开 IDE 或代码编辑器，通过系统自带的命令行直接启动 **`train.py`** 并展示训练日志。
  - **使用方法**：
    - 激活对应的 Python 虚拟环境后，双击运行 **`Run_train.bat`**。
    - 本脚本自带训练结束后的 **自动关机** 功能，通过一个名为 **`no_shutdown`** 的文件的 **位置** 来控制。
    - 当 `no_shutdown` 文件放置在 **项目根目录** 下（即与 `Run_train.bat` 同级）时，自动关机功能 **开启**；当该文件被移至其他位置或重命名时，自动关机功能 **关闭**。

## 性能评估 / Evaluation

- **`logs_copy.py`**
  - **功能**：批量收集并迁移分散的训练日志，防止同名覆盖。
  - **使用方法**：在程序主入口配置好实验结果源目录和集中输出路径（均有预设默认值），运行代码 **`logs_copy.py`** 即可。

- **`draw_TV_curve.py`**
  - **功能**：绘制 Train Loss 和 Val Loss 随着 Epoch 变化的平滑对比曲线。
  - **使用方法**：在程序主入口填写好相关配置（颜色映射、平滑算法、读取和保存地址等均有预设默认值），运行代码 **`draw_TV_curve.py`** 即可。

- **`image_compare.py`**
  - **功能**：绘制不同模型对同一图像的预测结果对照图网格（定性分析）。
  - **使用方法**：在程序主入口填写好相关配置（待对比的模型配置字典、所需提取的图片 ID 列表等），运行代码 **`image_compare.py`** 即可。

- **`metrics_evaluate.py`**
  - **功能**：计算不同模型对测试集的各项分割与分类指标，并生成 CSV 定量分析表格。
  - **使用方法**：在程序主入口填写好需要评估的模型配置字典与保存路径，运行代码 **`metrics_evaluate.py`** 即可自动追加数据。

- **`metrics_plot.py`**
  - **功能**：根据生成的各项指标 CSV 表格文件，生成带有标准差置信区间的可视化柱状图（定量分析）。
  - **使用方法**：在程序主入口选择对应的分析模式（Loss / Optimizer / Ablation），运行代码 **`metrics_plot.py`** 即可。

---

# **Few-Shot Skin Lesion Medical Image Segmentation Toolkit / FSL-SkinSeg-Toolkit**

- This project is modified, refined, and organized based on my graduation project code. As an ongoing project, there may still be minor issues and room for improvement. User feedback and issue reports are highly welcome.
- By default, this project trains models using the **PH2 dataset**, so the data loading module is specifically tailored to its directory structure. If you wish to use other datasets, please modify the corresponding configurations.
- Due to personal hardware constraints, this project was originally built for a pure CPU training environment. Therefore, many underlying algorithms (e.g., loss function calculations) have been deeply optimized and adapted for **CPU-only computation**.

**Note**: All paths mentioned below are relative to the project root directory (**FSL-SkinImgSeg-Toolkit**).

# Guide

## Preprocess

- **`datasets_split.py`**
  - **Function**: Randomly splits the original dataset into training, validation, and testing sets at a ratio of 6:2:2.
  - **Usage**:
    - Taking the PH2 dataset as an example, place the extracted folders (where raw images and masks share the same directory) into the `\images` folder. The structure should look like this:
      - `images`
        - `IMD003`
          - `IMD003_Dermoscopic_Image`
          - `IMD003_lesion`
          - `IMD003_roi`
        - `IMD004` (ditto)
    - Once the `images` folder is ready, simply run **`datasets_split.py`**.
    - **Note**: You do not need to manually create the `images` and `images_split` folders before running the program; they will be generated automatically.

- **`image_to_pt.py`**
  - **Function**: Converts image data into PyTorch `.pt` format files to accelerate loading speeds, trading storage space for training time.
  - **Usage**: Select the corresponding parameters in the Python main entry point (inside `if __name__ == '__main__':`) and run **`image_to_pt.py`**.

## Train

- **`my_dataset.py` / `my_model.py` / `losses.py`**
  - **Function**: Basic core components corresponding to the dataset class, the basic U-Net model architecture, and the custom loss function library (including the proprietary `TverskyHausdorffLoss`) used during training.

- **`train.py`**
  - **Function**: The core main program executing the model training pipeline.
  - **Usage**:
    - First, go to the task queue configuration file **`logs\tasks.yml`** to set up the experimental parameters you need to run.
    - After configuration, double-click **`Run_train.bat`** (assuming your Python environment is properly configured), or run **`train.py`** directly in the terminal to start automated training.
    - After reading the tasks in **`logs\tasks.yml`**, the program will pop them from the queue and archive all training histories and timestamps into **`logs\tasks_history.yml`**.
    - **Note**: If `tasks.yml` does not exist before your first run, simply run the program once, and a template file will be generated automatically. You can modify it according to the built-in examples later.

- **`Run_train.bat`** (Windows Only)
  - **Function**: Starts **`train.py`** and displays training logs directly via the command line without needing to open an IDE or code editor.
  - **Usage**:
    - After activating the target Python virtual environment, double-click **`Run_train.bat`**.
    - This script includes an **Auto-Shutdown** feature triggered after training finishes, controlled by the **location** of a file named **`no_shutdown`**.
    - When `no_shutdown` is placed in the **project root directory** (same level as `Run_train.bat`), the auto-shutdown feature is **enabled**. If the file is moved elsewhere or renamed, the feature is **disabled**.

## Evaluation

- **`logs_copy.py`**
  - **Function**: Batch collects and migrates scattered training logs while preventing file overwrite conflicts.
  - **Usage**: Configure the source directory of the experimental results and the centralized output path in the main entry point (defaults provided), then run **`logs_copy.py`**.

- **`draw_TV_curve.py`**
  - **Function**: Draws smoothed comparison curves of Train Loss and Val Loss across epochs.
  - **Usage**: Fill in the relevant configurations (color mapping, smoothing algorithm, read/save paths, etc., are all pre-configured with defaults) in the main entry point, then run **`draw_TV_curve.py`**.

- **`image_compare.py`**
  - **Function**: Generates side-by-side comparison grid images of different models predicting the same input (Qualitative Analysis).
  - **Usage**: Configure the relevant settings (e.g., the model configuration dictionary to compare, target image ID lists) in the main entry point, then run **`image_compare.py`**.

- **`metrics_evaluate.py`**
  - **Function**: Calculates various segmentation and classification metrics of different models on the test set, generating a CSV table for quantitative analysis.
  - **Usage**: Fill in the model configuration dictionary and save path to be evaluated in the main entry point, then run **`metrics_evaluate.py`** to automatically append the new data.

- **`metrics_plot.py`**
  - **Function**: Generates visual bar charts with standard deviation confidence intervals based on the generated metrics CSV files (Quantitative Analysis).
  - **Usage**: Select the corresponding analysis mode (Loss / Optimizer / Ablation) in the main entry point, then run **`metrics_plot.py`**.