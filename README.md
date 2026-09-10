# **Few-Shot Skin Lesion Medical Image Segmentation Toolkit / FSL-SkinSeg-Toolkit**

- [简体中文](./README_CN.md) | [English] 

# Introduction

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
