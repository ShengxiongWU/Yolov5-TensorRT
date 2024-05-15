#include "cmd_line_parser.h"
#include "engine.h"
#include <chrono>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>
#include <dirent.h>

float NMS_THRESHOLD = 0.1;
float confidence_threshold = 0.9;
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

float test(Precision p, CommandLineArguments arguments){



    // Specify our GPU inference configuration options
    Options options;
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    options.precision = p;
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


    cv::RNG rng(12345);

    // 为每个类别生成一个随机颜色并存储在map中
    std::map<int, cv::Scalar> colors;
    for (size_t i = 0; i < class_names.size(); ++i) {
        colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    std::string input_path = "../testInput";
    std::string output_path = "../outputs"; // 设置输出目录
    std::vector<cv::String> filenames;
    std::vector<cv::String> tf;
    filenames.clear();

    cv::glob(input_path + "/*.jpg", filenames);

    cv::glob(input_path + "/*.png", tf);
    
    for(const auto& s : tf){
        filenames.emplace_back(s);
    }

    float avgTotalTime = 0;
    size_t counter = 0;

    const auto &inputDims = engine.getInputDims();
    for (const auto& filename : filenames) {
        // std::cout<<filename<<std::endl;
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

            size_t WarmingIteration = 1000;
            size_t WarmingIterationNew = 500;
            bool first = false;
            if(first){
                WarmingIteration = WarmingIterationNew;
            }
            // Warm up the network before we begin the benchmark
            // std::cout << "\nWarming up the network..." << std::endl;
            std::vector<std::vector<std::vector<float>>> featureVectors;
            for (size_t i = 0; i < WarmingIteration; ++i) {
                bool succ = engine.runInference(inputs, featureVectors);
                if (!succ) {
                    throw std::runtime_error("Unable to run inference.");
                }
            }

            // Benchmark the inference time
            size_t numIterations = 1000;
            // std::cout << "Warmup done. Running benchmarks (" << numIterations << " iterations)...\n" << std::endl;
            preciseStopwatch stopwatch;
            for (size_t i = 0; i < numIterations; ++i) {
                featureVectors.clear();
                engine.runInference(inputs, featureVectors);
            }
            auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
            auto avgElapsedTimeMs = totalElapsedTimeMs / numIterations / static_cast<float>(inputs[0].size());

            avgTotalTime += avgElapsedTimeMs;
            counter++;

            // std::cout << input_path + " Benchmarking complete!" << std::endl;
            // std::cout << "======================" << std::endl;
            // std::cout << "Avg time per sample: " << std::endl;
            // std::cout << avgElapsedTimeMs << " ms" << std::endl;
            // std::cout << "Batch size: " << std::endl;
            // std::cout << inputs[0].size() << std::endl;
            // std::cout << "Avg FPS: " << std::endl;
            // std::cout << static_cast<int>(1000 / avgElapsedTimeMs) << " fps" << std::endl;
            // std::cout << "======================\n" << std::endl;

        }
    }
    return avgTotalTime/counter;
}

int main(int argc, char *argv[]) {
    CommandLineArguments arguments;

    // Parse the command line arguments
    if (!parseArguments(argc, argv, arguments)) {
        return -1;
    }



    float avgTotalTimeFP32; 
    float avgTotalTimeFP16; 
    float avgTotalTimeINT8; 

    avgTotalTimeFP32 = test(Precision::FP32, arguments);
    avgTotalTimeFP16 = test(Precision::FP16, arguments);
    avgTotalTimeINT8 = test(Precision::INT8, arguments);
    std::cout << "FP32 Benchmarking complete!" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Total avg time per sample: " << std::endl;
    std::cout << avgTotalTimeFP32 << " ms" << std::endl;
    std::cout << "Total Avg FPS: " << std::endl;
    std::cout << static_cast<int>(1000 / avgTotalTimeFP32) << " fps" << std::endl;
    std::cout << "======================\n" << std::endl;
    
    std::cout << "FP16 Benchmarking complete!" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Total avg time per sample: " << std::endl;
    std::cout << avgTotalTimeFP16 << " ms" << std::endl;
    std::cout << "Total Avg FPS: " << std::endl;
    std::cout << static_cast<int>(1000 / avgTotalTimeFP16) << " fps" << std::endl;
    std::cout << "======================\n" << std::endl;

    std::cout << "INT8 Benchmarking complete!" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Total avg time per sample: " << std::endl;
    std::cout << avgTotalTimeINT8 << " ms" << std::endl;
    std::cout << "Total Avg FPS: " << std::endl;
    std::cout << static_cast<int>(1000 / avgTotalTimeINT8) << " fps" << std::endl;
    std::cout << "======================\n" << std::endl;


    return 0;
}


