# Yolov5-TensorRT

基于 [cyrusbehr/tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) 编写的Yolov5 TensorRT推理加速。

使用方法和原仓库基本相同，`run_inference` 会使用 `main.cpp` 中选择的精度对 `inputs` 文件夹下的所有 PNG 和 JPG 图片进行推理，推理结果会输出到 `outputs` 文件夹中。

## 环境要求

与原仓库相同，以下部分为原仓库环境要求的中文翻译版：

- 在 Ubuntu 20.04 和 22.04 上经过测试，目前 **不支持** Windows。
- 安装 CUDA 11 或 12，具体指南参见 [此处](https://developer.nvidia.com/cuda-downloads)。
  - 推荐版本 >= 12.0
  - 最低要求版本 >= 11.0
- 安装 cuDNN，具体指南参见 [此处](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download)。
  - 最低要求版本 >= 8
  - 最高要求版本 < 9（因为 OpenCV GPU 目前尚不支持）
- `sudo apt install build-essential`
- `sudo snap install cmake --classic`
- 使用 CUDA 支持编译安装 OpenCV。要从源代码编译 OpenCV，请运行 `./scripts/build_opencv.sh` 提供的脚本。
  - 如果使用提供的脚本，并且你已将 cuDNN 安装到非标准位置，则必须修改脚本中的 `CUDNN_INCLUDE_DIR` 和 `CUDNN_LIBRARY` 变量。
  - 推荐版本 >= 4.8
- 从 [此处](https://developer.nvidia.com/tensorrt/download/10x) 下载 TensorRT 10。
  - 最低要求版本 >= 10.0
- 导航到 `CMakeLists.txt` 文件，并将 `TODO` 替换为你的 TensorRT 安装路径。

## 使用方法

1. 将从 Yolov5 导出的 FP32 格式的输入 ONNX 文件放到 `models` 文件夹下。
2. 在 `tensorrt-cpp-api-main` 文件夹下使用以下命令构建项目：

    ```bash
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    ```
   
3. 在 `build` 文件夹中执行以下命令运行推理：

    ```bash
    ./run_inference --onnx_model ../models/yolov5n.onnx
    ```

    其中 `../models/yolov5n.onnx` 是要使用的模型，你可以将其替换为你想要使用的模型的名称。

## 修改精度的方法

- Float32：`options.precision = Precision::FP32;`
- Float16：`options.precision = Precision::FP16;`
- Int8：`options.precision = Precision::INT8;`

## 修改输入输出位置

- 改开头的的input_path和output_path

## 修改NMS参数

- 改开头的NMS_THRESHOLD， confidence_threshold 和 topK
