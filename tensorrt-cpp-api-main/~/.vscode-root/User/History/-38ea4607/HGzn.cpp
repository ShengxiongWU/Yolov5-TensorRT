#include "cmd_line_parser.h"
#include "engine.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <dirent.h>

float NMS_THRESHOLD = 0.1;
float confidence_threshold = 0.9;

int main(int argc, char *argv[]) {
    CommandLineArguments arguments;

    // Parse the command line arguments
    if (!parseArguments(argc, argv, arguments)) {
        return -1;
    }

    // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = Precision::FP16;
    // If using INT8 precision, must specify path to directory containing
    // calibration data.
    options.calibrationDataDirectoryPath = "/home/sxwu/yolov5/tensorrt-cpp-api-main/calibrationData/val2017";
    // Specify the batch size to optimize for.
    options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    options.maxBatchSize = 1;

    Engine<float> engine(options);

    // Define our preprocessing code
    // The default Engine::build method will normalize values between [0.f, 1.f]
    // Setting the normalize flag to false will leave values between [0.f, 255.f]
    // (some converted models may require this).

    // For our YoloV8 model, we need the values to be normalized between
    // [0.f, 1.f] so we use the following params
    std::array<float, 3> subVals{0.f, 0.f, 0.f};
    std::array<float, 3> divVals{1.f, 1.f, 1.f};
    bool normalize = true;
    // Note, we could have also used the default values.

    // If the model requires values to be normalized between [-1.f, 1.f], use the
    // following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;

    if (!arguments.onnxModelPath.empty()) {
        // Build the onnx model into a TensorRT engine file, and load the TensorRT
        // engine file into memory.
        bool succ = engine.buildLoadNetwork(arguments.onnxModelPath, subVals, divVals, normalize);
        if (!succ) {
            throw std::runtime_error("Unable to build or load TensorRT engine.");
        }
    } else {
        // Load the TensorRT engine file directly
        bool succ = engine.loadNetwork(arguments.trtModelPath, subVals, divVals, normalize);
        if (!succ) {
            throw std::runtime_error("Unable to load TensorRT engine.");
        }
    }


    std::vector<std::vector<std::vector<float>>> featureVectors;


    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    cv::RNG rng(12345);

    // 为每个类别生成一个随机颜色并存储在map中
    std::map<int, cv::Scalar> colors;
    for (size_t i = 0; i < class_names.size(); ++i) {
        colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    std::string input_path = "../inputs";
    std::string output_path = "../outputs"; // 设置输出目录
    std::vector<cv::String> filenames;
    std::vector<cv::String> tf;
    filenames.clear();

    cv::glob(input_path + "/*.jpg", filenames);

    cv::glob(input_path + "/*.png", tf);
    
    for(const auto& s : tf){
        filenames.emplace_back(s);
    }

    const auto &inputDims = engine.getInputDims();
    for (const auto& filename : filenames) {
        std::cout<<filename<<std::endl;
        cv::Mat image = cv::imread(filename);
        if (!image.empty()) {
            // Upload the image GPU memory
            cv::cuda::GpuMat img;
            img.upload(image);

            // The model expects RGB input
            cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

            // In the following section we populate the input vectors to later pass for
            // inference
            std::vector<std::vector<cv::cuda::GpuMat>> inputs;
            inputs.clear();
            // Let's use a batch size which matches that which we set the
            // Options.optBatchSize option
            size_t batchSize = options.optBatchSize;

            for (const auto &inputDim : inputDims) { // For each of the model inputs...
                std::vector<cv::cuda::GpuMat> input;
                for (size_t j = 0; j < batchSize; ++j) { // For each element we want to add to the batch...

                    auto resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);

                    input.emplace_back(std::move(resized));
                }
                inputs.emplace_back(std::move(input));
                input.clear();
            }
            std::cout << "\nstart run inference" << std::endl;
            bool succ = engine.runInference(inputs, featureVectors);
            if (!succ) {
                throw std::runtime_error("Unable to run inference.");
            }

        }
    }



    size_t i = 0;
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    for (const auto& filename : filenames) {
        cv::Mat image = cv::imread(filename);
        if(!image.empty()){    
            const std::vector<std::vector<float>> detections = featureVectors[i++];
            for (const std::vector<float>& detection : detections) {
                for(size_t i = 0; i < detection.size();i+=85){
                    // 解析检测结果
                    float cx = detection[i]; // 中心点x坐标
                    float cy = detection[i+1]; // 中心点y坐标
                    float w = detection[i+2];  // 宽度
                    float h = detection[i+3];  // 高度

                    // 转换中心点坐标为左上角坐标
                    float x1 = cx - w / 2;
                    float y1 = cy - h / 2;

                    // 获取最高的类别置信度和对应的类别索引
                    float max_class_score = 0;
                    int max_class_id = -1;
                    for (size_t j = i+4; j < i+85; j++) {
                        if (detection[j] > max_class_score) {
                            max_class_score = detection[j];
                            max_class_id = j - 5 - i; // 类别索引从0开始
                        }
                    }

                    // 只有当总置信度和类别置信度都高于阈值时才绘制边界框和类别标签
                    cv::Rect  box(x1, y1, w, h);
                    bboxes.push_back(box);
                    scores.push_back(max_class_score);
                    labels.push_back(max_class_id);
                
                }
            }
        
    

            
            std::vector<int> indices;
            cv::dnn::NMSBoxesBatched(bboxes, scores, labels confidence_threshold, NMS_THRESHOLD, indices);
            int k = 0;
            int topK = 10;
            for (int idx : indices) {
                    if(k++>=topK){
                        std::cout<<"here"<<std::endl;
                        break;
                    }
                    const auto& box = bboxes[idx];
                    int class_id = labels[idx]; // 获取类别ID
                    cv::Scalar color = colors[class_id]; // 获取对应类别的颜色

                    float r = std::min(640 / (image.cols * 1.0), 640 / (image.rows * 1.0));

                    float x = box.x / r;
                    float y = box.y / r;
                    float w = box.width / r;
                    float h = box.height / r;
                    // 考虑填充
                    cv::Rect2f mapped_box = cv::Rect2f(x, y, w, h);

                    // 绘制边界框
                    cv::rectangle(image, mapped_box, color, 3);
                    std::string label = class_names[labels[idx]] + ": " + std::to_string(scores[idx]);
                    cv::putText(image, label, cv::Point(mapped_box.x, mapped_box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }

            // 构造输出文件名
            std::string base_filename = filename.substr(filename.find_last_of("/\\") + 1);
            std::string output_filename = output_path + "/" + base_filename;

            // 保存处理后的图像到输出目录
            cv::imwrite(output_filename, image);

            bboxes.clear();
            scores.clear();
            labels.clear();
            indices.clear();
        }
    }



    return 0;
}


