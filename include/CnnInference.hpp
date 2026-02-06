/**
 * @file CnnInference.hpp
 * @brief CNN Inference Module for Layer 4 of Cascaded Framework
 * 
 * Triển khai inference với CNN (MobileNetV2/ShuffleNetV1) cho các mẫu 
 * ambiguous không thể phân loại bởi Random Forest layers.
 * 
 * Features:
 * - ONNX Runtime C++ API cho cross-platform inference
 * - INT8 quantization simulation cho FPGA compatibility
 * - Batch inference support
 * - Easy replacement với Vitis-AI Library cho Ultra96-V2
 * 
 * @author Research Team
 * @date 2026
 */

#ifndef CNN_INFERENCE_HPP
#define CNN_INFERENCE_HPP

#include "WaveletTransform.hpp"
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <functional>
#include <map>

namespace respiratory {

// Forward declaration for PIMPL pattern (hide ONNX Runtime details)
class CnnInferenceImpl;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Số classes đầu ra của CNN
constexpr int CNN_NUM_CLASSES = 4;

/// Kích thước input image (224x224 cho MobileNetV2)
constexpr int CNN_INPUT_SIZE = 224;

/// Số channels input (1 cho grayscale spectrogram)
constexpr int CNN_INPUT_CHANNELS = 1;

// ============================================================================
// ENUMS
// ============================================================================

/**
 * @enum CnnModelType
 * @brief Loại model CNN được hỗ trợ
 */
enum class CnnModelType {
    MOBILENET_V2,       ///< MobileNetV2 (~4.4M params)
    SHUFFLENET_V1,      ///< ShuffleNetV1 (~3.5M params)
    CUSTOM              ///< Custom model từ file ONNX
};

/**
 * @enum InferencePrecision
 * @brief Precision cho inference
 */
enum class InferencePrecision {
    FLOAT32,            ///< Full precision (FP32)
    FLOAT16,            ///< Half precision (FP16)
    INT8                ///< 8-bit quantized (cho FPGA DPU)
};

/**
 * @enum ExecutionProvider
 * @brief Backend cho inference
 */
enum class ExecutionProvider {
    CPU,                ///< CPU inference
    CUDA,               ///< NVIDIA GPU (nếu có)
    TENSORRT,           ///< TensorRT optimization
    DPU,                ///< Xilinx DPU (Ultra96-V2)
    OPENVINO            ///< Intel OpenVINO
};

/**
 * @enum CnnPredictionClass
 * @brief Các class dự đoán của CNN
 */
enum class CnnPredictionClass : int {
    NORMAL = 0,
    CRACKLE = 1,
    WHEEZE = 2,
    BOTH = 3,
    UNKNOWN = -1
};

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @struct CnnConfig
 * @brief Cấu hình cho CNN inference
 */
struct CnnConfig {
    CnnModelType modelType = CnnModelType::MOBILENET_V2;
    InferencePrecision precision = InferencePrecision::FLOAT32;
    ExecutionProvider provider = ExecutionProvider::CPU;
    
    std::string modelPath;              ///< Đường dẫn file ONNX model
    int inputWidth = CNN_INPUT_SIZE;    
    int inputHeight = CNN_INPUT_SIZE;
    int inputChannels = CNN_INPUT_CHANNELS;
    int numClasses = CNN_NUM_CLASSES;
    
    int batchSize = 1;                  ///< Batch size (1 cho real-time)
    int numThreads = 4;                 ///< Số threads cho CPU inference
    
    bool enableProfiling = false;       ///< Bật profiling cho optimization
    bool enableMemoryArena = true;      ///< Tối ưu memory allocation
    
    CnnConfig() = default;
};

/**
 * @struct CnnPrediction
 * @brief Kết quả dự đoán từ CNN
 */
struct CnnPrediction {
    CnnPredictionClass predictedClass;  ///< Class dự đoán
    float confidence;                    ///< Độ tin cậy (0-1)
    std::array<float, CNN_NUM_CLASSES> probabilities;  ///< Xác suất từng class
    
    float inferenceTimeMs;               ///< Thời gian inference (ms)
    bool isValid;                        ///< Kết quả có hợp lệ?
    
    CnnPrediction() : predictedClass(CnnPredictionClass::UNKNOWN),
                      confidence(0.0f), inferenceTimeMs(0.0f), isValid(false) {
        probabilities.fill(0.0f);
    }
    
    /**
     * @brief Lấy tên class dạng string
     */
    std::string getClassName() const;
    
    /**
     * @brief Mô tả kết quả
     */
    std::string describe() const;
};

/**
 * @struct CnnStatistics
 * @brief Thống kê inference
 */
struct CnnStatistics {
    int totalInferences = 0;
    int successfulInferences = 0;
    float totalInferenceTimeMs = 0.0f;
    float avgInferenceTimeMs = 0.0f;
    float minInferenceTimeMs = 0.0f;
    float maxInferenceTimeMs = 0.0f;
    
    std::array<int, CNN_NUM_CLASSES> classCounts = {0};
    
    void update(const CnnPrediction& pred);
    void reset();
    std::string describe() const;
};

/**
 * @struct ModelInfo
 * @brief Thông tin về model đã load
 */
struct ModelInfo {
    std::string name;
    std::string version;
    
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;
    
    int64_t numParameters = 0;
    float modelSizeMB = 0.0f;
    
    bool isLoaded = false;
};

// ============================================================================
// CNN INFERENCE CLASS
// ============================================================================

/**
 * @class CnnInference
 * @brief Module inference CNN cho Layer 4
 * 
 * Pipeline:
 * 1. Nhận spectrogram từ WaveletTransform
 * 2. Preprocess (resize, normalize nếu cần)
 * 3. Chạy inference với ONNX Runtime
 * 4. Postprocess và trả về prediction
 * 
 * Thiết kế để dễ dàng thay thế bằng Vitis-AI:
 * - Interface rõ ràng
 * - Input/output format chuẩn
 * - Có thể wrap DPU API vào cùng interface
 */
class CnnInference {
public:
    // ========================================================================
    // LIFECYCLE
    // ========================================================================
    
    /**
     * @brief Constructor
     */
    CnnInference();
    
    /**
     * @brief Constructor với config
     */
    explicit CnnInference(const CnnConfig& config);
    
    /**
     * @brief Destructor
     */
    ~CnnInference();
    
    // Disable copy (do ONNX session không thể copy)
    CnnInference(const CnnInference&) = delete;
    CnnInference& operator=(const CnnInference&) = delete;
    
    // Enable move
    CnnInference(CnnInference&&) noexcept;
    CnnInference& operator=(CnnInference&&) noexcept;
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Load model từ file ONNX
     * @param modelPath Đường dẫn file .onnx
     * @return true nếu load thành công
     */
    bool loadModel(const std::string& modelPath);
    
    /**
     * @brief Load model với config đầy đủ
     */
    bool loadModel(const CnnConfig& config);
    
    /**
     * @brief Kiểm tra model đã load chưa
     */
    bool isModelLoaded() const { return m_isLoaded; }
    
    /**
     * @brief Lấy thông tin model
     */
    const ModelInfo& getModelInfo() const { return m_modelInfo; }
    
    // ========================================================================
    // MAIN INFERENCE INTERFACE
    // ========================================================================
    
    /**
     * @brief Chạy inference trên spectrogram
     * 
     * @param spectrogram Spectrogram đầu vào (đã normalize)
     * @return Kết quả prediction
     */
    CnnPrediction predict(const Spectrogram& spectrogram);
    
    /**
     * @brief Chạy inference trên raw audio signal
     * 
     * Tự động tạo spectrogram sử dụng WaveletTransform
     * 
     * @param signal Tín hiệu audio đầu vào
     * @param sampleRate Sample rate của signal
     * @return Kết quả prediction
     */
    CnnPrediction predictFromSignal(const std::vector<float>& signal,
                                     int sampleRate = WAVELET_SAMPLE_RATE);
    
    /**
     * @brief Chạy inference trên tensor input
     * 
     * @param input Vector float đã format đúng (NCHW hoặc NHWC)
     * @return Kết quả prediction
     */
    CnnPrediction predictFromTensor(const std::vector<float>& input);
    
    // ========================================================================
    // BATCH INFERENCE
    // ========================================================================
    
    /**
     * @brief Batch inference trên nhiều spectrograms
     */
    std::vector<CnnPrediction> predictBatch(
        const std::vector<Spectrogram>& spectrograms);
    
    /**
     * @brief Batch inference từ signals
     */
    std::vector<CnnPrediction> predictBatchFromSignals(
        const std::vector<std::vector<float>>& signals,
        int sampleRate = WAVELET_SAMPLE_RATE);
    
    // ========================================================================
    // CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Đặt config mới
     */
    void setConfig(const CnnConfig& config);
    
    /**
     * @brief Lấy config hiện tại
     */
    const CnnConfig& getConfig() const { return m_config; }
    
    /**
     * @brief Đặt số threads
     */
    void setNumThreads(int numThreads);
    
    /**
     * @brief Đặt precision
     */
    void setPrecision(InferencePrecision precision);
    
    /**
     * @brief Bật/tắt profiling
     */
    void enableProfiling(bool enable);
    
    // ========================================================================
    // STATISTICS & PROFILING
    // ========================================================================
    
    /**
     * @brief Lấy thống kê inference
     */
    const CnnStatistics& getStatistics() const { return m_stats; }
    
    /**
     * @brief Reset thống kê
     */
    void resetStatistics() { m_stats.reset(); }
    
    /**
     * @brief Warmup model (chạy vài inference để optimize)
     */
    void warmup(int numRuns = 5);
    
    // ========================================================================
    // VITIS-AI COMPATIBILITY INTERFACE
    // ========================================================================
    
    /**
     * @brief Callback type cho custom inference engine (Vitis-AI, etc.)
     * 
     * Cho phép inject custom inference logic mà không thay đổi interface
     */
    using CustomInferenceCallback = std::function<
        std::vector<float>(const std::vector<float>& input)>;
    
    /**
     * @brief Đặt custom inference callback
     * 
     * Khi set, sẽ sử dụng callback này thay vì ONNX Runtime
     */
    void setCustomInferenceCallback(CustomInferenceCallback callback);
    
    /**
     * @brief Check xem đang dùng custom inference không
     */
    bool isUsingCustomInference() const { return m_customCallback != nullptr; }

private:
    CnnConfig m_config;
    ModelInfo m_modelInfo;
    CnnStatistics m_stats;
    
    bool m_isLoaded;
    
    // PIMPL for ONNX Runtime implementation
    std::unique_ptr<CnnInferenceImpl> m_impl;
    
    // Wavelet transform for signal-to-spectrogram conversion
    std::unique_ptr<WaveletTransform> m_waveletTransform;
    
    // Custom inference callback (for Vitis-AI integration)
    CustomInferenceCallback m_customCallback;
    
    // ========================================================================
    // PRIVATE METHODS
    // ========================================================================
    
    /**
     * @brief Preprocess spectrogram cho model input
     */
    std::vector<float> preprocessSpectrogram(const Spectrogram& spec);
    
    /**
     * @brief Postprocess model output thành prediction
     */
    CnnPrediction postprocessOutput(const std::vector<float>& output);
    
    /**
     * @brief Softmax function
     */
    std::vector<float> softmax(const std::vector<float>& logits);
    
    /**
     * @brief Chạy inference với ONNX Runtime
     */
    std::vector<float> runOnnxInference(const std::vector<float>& input);
    
    /**
     * @brief Chạy inference với custom callback
     */
    std::vector<float> runCustomInference(const std::vector<float>& input);
    
    /**
     * @brief Fallback simulation khi không có model
     */
    CnnPrediction simulateInference(const Spectrogram& spec);
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Chuyển đổi CnnPredictionClass sang string
 */
std::string cnnPredictionClassToString(CnnPredictionClass cls);

/**
 * @brief Chuyển đổi CnnModelType sang string
 */
std::string cnnModelTypeToString(CnnModelType type);

/**
 * @brief Tạo CnnInference với MobileNetV2 preset
 */
CnnInference createMobileNetV2Inference(const std::string& modelPath);

/**
 * @brief Tạo CnnInference cho Vitis-AI DPU
 * 
 * Trả về instance với custom callback placeholder
 * Cần implement callback với Vitis-AI Library
 */
CnnInference createDpuInference(const std::string& xmodelPath);

/**
 * @brief Validate model file
 */
bool validateOnnxModel(const std::string& modelPath);

/**
 * @brief Get supported execution providers
 */
std::vector<ExecutionProvider> getSupportedProviders();

} // namespace respiratory

#endif // CNN_INFERENCE_HPP
