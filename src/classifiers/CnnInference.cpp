/**
 * @file CnnInference.cpp
 * @brief Implementation of CNN Inference Module for Layer 4
 * 
 * Triển khai inference với CNN sử dụng:
 * - ONNX Runtime C++ API (khi có model)
 * - Simulation mode (khi chưa có model - cho development)
 * - Custom callback interface (cho Vitis-AI integration)
 * 
 * @author Research Team
 * @date 2026
 */

#include "CnnInference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <random>

// Conditional ONNX Runtime include
// Uncomment when ONNX Runtime is available
// #define USE_ONNX_RUNTIME
#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace respiratory {

// ============================================================================
// PIMPL IMPLEMENTATION CLASS
// ============================================================================

/**
 * @class CnnInferenceImpl
 * @brief PIMPL class hiding ONNX Runtime details
 */
class CnnInferenceImpl {
public:
    CnnInferenceImpl() = default;
    ~CnnInferenceImpl() = default;
    
#ifdef USE_ONNX_RUNTIME
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    
    Ort::AllocatorWithDefaultOptions allocator;
#endif
    
    bool isInitialized = false;
};

// ============================================================================
// UTILITY FUNCTIONS IMPLEMENTATION
// ============================================================================

std::string cnnPredictionClassToString(CnnPredictionClass cls) {
    switch (cls) {
        case CnnPredictionClass::NORMAL: return "Normal";
        case CnnPredictionClass::CRACKLE: return "Crackle";
        case CnnPredictionClass::WHEEZE: return "Wheeze";
        case CnnPredictionClass::BOTH: return "Both";
        default: return "Unknown";
    }
}

std::string cnnModelTypeToString(CnnModelType type) {
    switch (type) {
        case CnnModelType::MOBILENET_V2: return "MobileNetV2";
        case CnnModelType::SHUFFLENET_V1: return "ShuffleNetV1";
        case CnnModelType::CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

// ============================================================================
// CNN PREDICTION IMPLEMENTATION
// ============================================================================

std::string CnnPrediction::getClassName() const {
    return cnnPredictionClassToString(predictedClass);
}

std::string CnnPrediction::describe() const {
    std::ostringstream oss;
    oss << "CNN Prediction: " << getClassName()
        << " (confidence: " << std::fixed << std::setprecision(3) << confidence << ")"
        << ", inference time: " << inferenceTimeMs << " ms";
    
    oss << "\n  Probabilities: [";
    for (int i = 0; i < CNN_NUM_CLASSES; ++i) {
        oss << cnnPredictionClassToString(static_cast<CnnPredictionClass>(i))
            << "=" << std::setprecision(3) << probabilities[i];
        if (i < CNN_NUM_CLASSES - 1) oss << ", ";
    }
    oss << "]";
    
    return oss.str();
}

// ============================================================================
// CNN STATISTICS IMPLEMENTATION
// ============================================================================

void CnnStatistics::update(const CnnPrediction& pred) {
    totalInferences++;
    
    if (pred.isValid) {
        successfulInferences++;
        
        totalInferenceTimeMs += pred.inferenceTimeMs;
        avgInferenceTimeMs = totalInferenceTimeMs / successfulInferences;
        
        if (minInferenceTimeMs == 0.0f || pred.inferenceTimeMs < minInferenceTimeMs) {
            minInferenceTimeMs = pred.inferenceTimeMs;
        }
        if (pred.inferenceTimeMs > maxInferenceTimeMs) {
            maxInferenceTimeMs = pred.inferenceTimeMs;
        }
        
        int classIdx = static_cast<int>(pred.predictedClass);
        if (classIdx >= 0 && classIdx < CNN_NUM_CLASSES) {
            classCounts[classIdx]++;
        }
    }
}

void CnnStatistics::reset() {
    totalInferences = 0;
    successfulInferences = 0;
    totalInferenceTimeMs = 0.0f;
    avgInferenceTimeMs = 0.0f;
    minInferenceTimeMs = 0.0f;
    maxInferenceTimeMs = 0.0f;
    classCounts.fill(0);
}

std::string CnnStatistics::describe() const {
    std::ostringstream oss;
    oss << "CNN Inference Statistics:\n"
        << "  Total inferences: " << totalInferences << "\n"
        << "  Successful: " << successfulInferences << "\n"
        << "  Avg time: " << std::fixed << std::setprecision(2) << avgInferenceTimeMs << " ms\n"
        << "  Min time: " << minInferenceTimeMs << " ms\n"
        << "  Max time: " << maxInferenceTimeMs << " ms\n"
        << "  Class distribution:\n";
    
    for (int i = 0; i < CNN_NUM_CLASSES; ++i) {
        oss << "    " << cnnPredictionClassToString(static_cast<CnnPredictionClass>(i))
            << ": " << classCounts[i] << "\n";
    }
    
    return oss.str();
}

// ============================================================================
// CNN INFERENCE IMPLEMENTATION
// ============================================================================

CnnInference::CnnInference()
    : m_isLoaded(false)
    , m_impl(std::make_unique<CnnInferenceImpl>())
{
    // Initialize wavelet transform với default config
    WaveletConfig waveletConfig;
    waveletConfig.outputWidth = CNN_INPUT_SIZE;
    waveletConfig.outputHeight = CNN_INPUT_SIZE;
    waveletConfig.numChannels = CNN_INPUT_CHANNELS;
    m_waveletTransform = std::make_unique<WaveletTransform>(waveletConfig);
}

CnnInference::CnnInference(const CnnConfig& config)
    : m_config(config)
    , m_isLoaded(false)
    , m_impl(std::make_unique<CnnInferenceImpl>())
{
    WaveletConfig waveletConfig;
    waveletConfig.outputWidth = config.inputWidth;
    waveletConfig.outputHeight = config.inputHeight;
    waveletConfig.numChannels = config.inputChannels;
    m_waveletTransform = std::make_unique<WaveletTransform>(waveletConfig);
    
    if (!config.modelPath.empty()) {
        loadModel(config.modelPath);
    }
}

CnnInference::~CnnInference() = default;

CnnInference::CnnInference(CnnInference&&) noexcept = default;
CnnInference& CnnInference::operator=(CnnInference&&) noexcept = default;

bool CnnInference::loadModel(const std::string& modelPath) {
    CnnConfig config = m_config;
    config.modelPath = modelPath;
    return loadModel(config);
}

bool CnnInference::loadModel(const CnnConfig& config) {
    m_config = config;
    
#ifdef USE_ONNX_RUNTIME
    try {
        // Create ONNX Runtime environment
        m_impl->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CnnInference");
        
        // Session options
        m_impl->sessionOptions = std::make_unique<Ort::SessionOptions>();
        m_impl->sessionOptions->SetIntraOpNumThreads(config.numThreads);
        m_impl->sessionOptions->SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (config.enableMemoryArena) {
            m_impl->sessionOptions->EnableMemPattern();
            m_impl->sessionOptions->EnableCpuMemArena();
        }
        
        // Create session
        #ifdef _WIN32
        std::wstring wideModelPath(config.modelPath.begin(), config.modelPath.end());
        m_impl->session = std::make_unique<Ort::Session>(
            *m_impl->env, wideModelPath.c_str(), *m_impl->sessionOptions);
        #else
        m_impl->session = std::make_unique<Ort::Session>(
            *m_impl->env, config.modelPath.c_str(), *m_impl->sessionOptions);
        #endif
        
        // Get input/output info
        size_t numInputs = m_impl->session->GetInputCount();
        size_t numOutputs = m_impl->session->GetOutputCount();
        
        m_modelInfo.inputNames.clear();
        m_modelInfo.outputNames.clear();
        m_modelInfo.inputShapes.clear();
        m_modelInfo.outputShapes.clear();
        
        for (size_t i = 0; i < numInputs; ++i) {
            auto inputName = m_impl->session->GetInputNameAllocated(i, m_impl->allocator);
            m_modelInfo.inputNames.push_back(inputName.get());
            m_impl->inputNames.push_back(m_modelInfo.inputNames.back().c_str());
            
            auto inputInfo = m_impl->session->GetInputTypeInfo(i);
            auto tensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
            m_modelInfo.inputShapes.push_back(tensorInfo.GetShape());
        }
        
        for (size_t i = 0; i < numOutputs; ++i) {
            auto outputName = m_impl->session->GetOutputNameAllocated(i, m_impl->allocator);
            m_modelInfo.outputNames.push_back(outputName.get());
            m_impl->outputNames.push_back(m_modelInfo.outputNames.back().c_str());
            
            auto outputInfo = m_impl->session->GetOutputTypeInfo(i);
            auto tensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
            m_modelInfo.outputShapes.push_back(tensorInfo.GetShape());
        }
        
        m_modelInfo.name = cnnModelTypeToString(config.modelType);
        m_modelInfo.isLoaded = true;
        m_impl->isInitialized = true;
        m_isLoaded = true;
        
        std::cout << "[CnnInference] Model loaded successfully: " << config.modelPath << "\n";
        std::cout << "  Inputs: " << numInputs << ", Outputs: " << numOutputs << "\n";
        
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[CnnInference] ONNX Runtime error: " << e.what() << "\n";
        return false;
    }
#else
    // Simulation mode - no actual ONNX Runtime
    std::cout << "[CnnInference] Running in SIMULATION mode (ONNX Runtime not linked)\n";
    std::cout << "  Model path: " << config.modelPath << "\n";
    
    m_modelInfo.name = cnnModelTypeToString(config.modelType);
    m_modelInfo.version = "simulation";
    m_modelInfo.inputNames = {"input"};
    m_modelInfo.outputNames = {"output"};
    m_modelInfo.inputShapes = {{1, CNN_INPUT_CHANNELS, CNN_INPUT_SIZE, CNN_INPUT_SIZE}};
    m_modelInfo.outputShapes = {{1, CNN_NUM_CLASSES}};
    m_modelInfo.numParameters = (config.modelType == CnnModelType::MOBILENET_V2) ? 
                                 4400000 : 3500000;
    m_modelInfo.modelSizeMB = m_modelInfo.numParameters * 4.0f / (1024 * 1024);
    m_modelInfo.isLoaded = true;
    
    m_isLoaded = true;  // Set to true for simulation mode
    return true;
#endif
}

CnnPrediction CnnInference::predict(const Spectrogram& spectrogram) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    CnnPrediction prediction;
    
    // Check if custom callback is set
    if (m_customCallback) {
        auto input = preprocessSpectrogram(spectrogram);
        auto output = runCustomInference(input);
        prediction = postprocessOutput(output);
    }
    else if (m_isLoaded) {
#ifdef USE_ONNX_RUNTIME
        auto input = preprocessSpectrogram(spectrogram);
        auto output = runOnnxInference(input);
        prediction = postprocessOutput(output);
#else
        // Simulation mode
        prediction = simulateInference(spectrogram);
#endif
    }
    else {
        // No model loaded - use simulation
        prediction = simulateInference(spectrogram);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    prediction.inferenceTimeMs = std::chrono::duration<float, std::milli>(
        endTime - startTime).count();
    prediction.isValid = true;
    
    m_stats.update(prediction);
    
    return prediction;
}

CnnPrediction CnnInference::predictFromSignal(const std::vector<float>& signal,
                                               int sampleRate) {
    // Create spectrogram from signal
    WaveletConfig config = m_waveletTransform->getConfig();
    config.sampleRate = sampleRate;
    m_waveletTransform->setConfig(config);
    
    Spectrogram spectrogram;
    if (!m_waveletTransform->transform(signal, spectrogram)) {
        CnnPrediction failPred;
        failPred.predictedClass = CnnPredictionClass::UNKNOWN;
        failPred.isValid = false;
        return failPred;
    }
    
    return predict(spectrogram);
}

CnnPrediction CnnInference::predictFromTensor(const std::vector<float>& input) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    CnnPrediction prediction;
    
    if (m_customCallback) {
        auto output = runCustomInference(input);
        prediction = postprocessOutput(output);
    }
    else {
#ifdef USE_ONNX_RUNTIME
        auto output = runOnnxInference(input);
        prediction = postprocessOutput(output);
#else
        // Simulation - just return dummy prediction
        prediction.predictedClass = CnnPredictionClass::NORMAL;
        prediction.confidence = 0.7f;
        prediction.probabilities = {0.7f, 0.1f, 0.1f, 0.1f};
#endif
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    prediction.inferenceTimeMs = std::chrono::duration<float, std::milli>(
        endTime - startTime).count();
    prediction.isValid = true;
    
    m_stats.update(prediction);
    
    return prediction;
}

std::vector<CnnPrediction> CnnInference::predictBatch(
    const std::vector<Spectrogram>& spectrograms) {
    
    std::vector<CnnPrediction> predictions;
    predictions.reserve(spectrograms.size());
    
    for (const auto& spec : spectrograms) {
        predictions.push_back(predict(spec));
    }
    
    return predictions;
}

std::vector<CnnPrediction> CnnInference::predictBatchFromSignals(
    const std::vector<std::vector<float>>& signals,
    int sampleRate) {
    
    std::vector<CnnPrediction> predictions;
    predictions.reserve(signals.size());
    
    for (const auto& signal : signals) {
        predictions.push_back(predictFromSignal(signal, sampleRate));
    }
    
    return predictions;
}

void CnnInference::setConfig(const CnnConfig& config) {
    m_config = config;
}

void CnnInference::setNumThreads(int numThreads) {
    m_config.numThreads = numThreads;
}

void CnnInference::setPrecision(InferencePrecision precision) {
    m_config.precision = precision;
}

void CnnInference::enableProfiling(bool enable) {
    m_config.enableProfiling = enable;
}

void CnnInference::warmup(int numRuns) {
    std::cout << "[CnnInference] Warming up with " << numRuns << " runs...\n";
    
    // Create dummy spectrogram
    Spectrogram dummySpec;
    dummySpec.allocate(m_config.inputWidth, m_config.inputHeight, m_config.inputChannels);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (float& v : dummySpec.data) {
        v = dist(gen);
    }
    
    // Run predictions
    for (int i = 0; i < numRuns; ++i) {
        predict(dummySpec);
    }
    
    // Reset stats after warmup
    m_stats.reset();
    
    std::cout << "[CnnInference] Warmup complete.\n";
}

void CnnInference::setCustomInferenceCallback(CustomInferenceCallback callback) {
    m_customCallback = callback;
}

// ============================================================================
// PRIVATE METHODS
// ============================================================================

std::vector<float> CnnInference::preprocessSpectrogram(const Spectrogram& spec) {
    /**
     * Preprocess spectrogram for model input
     * 
     * Steps:
     * 1. Ensure correct size (resize if needed)
     * 2. Convert to NCHW format
     * 3. Apply normalization if needed
     */
    
    // Check if resize needed
    std::vector<float> data;
    
    if (spec.width == m_config.inputWidth && 
        spec.height == m_config.inputHeight &&
        spec.channels == m_config.inputChannels) {
        // No resize needed - just convert to NCHW
        data = spec.toNCHW();
    }
    else {
        // Need to resize (simplified - just use raw data)
        // In production, proper resize interpolation should be used
        data = spec.toNCHW();
    }
    
    return data;
}

CnnPrediction CnnInference::postprocessOutput(const std::vector<float>& output) {
    /**
     * Postprocess raw model output to prediction
     * 
     * Assumes output is logits [batch, num_classes]
     */
    
    CnnPrediction prediction;
    
    if (output.size() < static_cast<size_t>(CNN_NUM_CLASSES)) {
        prediction.predictedClass = CnnPredictionClass::UNKNOWN;
        prediction.isValid = false;
        return prediction;
    }
    
    // Apply softmax
    auto probs = softmax(output);
    
    // Copy probabilities
    for (int i = 0; i < CNN_NUM_CLASSES && i < static_cast<int>(probs.size()); ++i) {
        prediction.probabilities[i] = probs[i];
    }
    
    // Find max class
    auto maxIt = std::max_element(prediction.probabilities.begin(), 
                                   prediction.probabilities.end());
    int maxIdx = std::distance(prediction.probabilities.begin(), maxIt);
    
    prediction.predictedClass = static_cast<CnnPredictionClass>(maxIdx);
    prediction.confidence = *maxIt;
    prediction.isValid = true;
    
    return prediction;
}

std::vector<float> CnnInference::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    
    // Find max for numerical stability
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp and sum
    float sumExp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sumExp += probs[i];
    }
    
    // Normalize
    for (float& p : probs) {
        p /= sumExp;
    }
    
    return probs;
}

std::vector<float> CnnInference::runOnnxInference(const std::vector<float>& input) {
#ifdef USE_ONNX_RUNTIME
    if (!m_impl->isInitialized || !m_impl->session) {
        return {};
    }
    
    try {
        // Create input tensor
        std::vector<int64_t> inputShape = {1, CNN_INPUT_CHANNELS, 
                                            CNN_INPUT_SIZE, CNN_INPUT_SIZE};
        
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, const_cast<float*>(input.data()), input.size(),
            inputShape.data(), inputShape.size());
        
        // Run inference
        auto outputTensors = m_impl->session->Run(
            Ort::RunOptions{nullptr},
            m_impl->inputNames.data(), &inputTensor, 1,
            m_impl->outputNames.data(), m_impl->outputNames.size());
        
        // Get output
        auto& outputTensor = outputTensors[0];
        auto* outputData = outputTensor.GetTensorData<float>();
        auto outputInfo = outputTensor.GetTensorTypeAndShapeInfo();
        size_t outputSize = outputInfo.GetElementCount();
        
        std::vector<float> output(outputData, outputData + outputSize);
        return output;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[CnnInference] Inference error: " << e.what() << "\n";
        return {};
    }
#else
    (void)input;
    return {};
#endif
}

std::vector<float> CnnInference::runCustomInference(const std::vector<float>& input) {
    if (m_customCallback) {
        return m_customCallback(input);
    }
    return {};
}

CnnPrediction CnnInference::simulateInference(const Spectrogram& spec) {
    /**
     * Simulation mode for development when no model is available
     * 
     * Uses simple heuristics based on spectrogram statistics:
     * - High energy variance -> abnormal
     * - Concentrated energy in low freq -> crackle-like
     * - Concentrated energy in mid freq -> wheeze-like
     */
    
    CnnPrediction prediction;
    prediction.probabilities.fill(0.1f);  // Base probabilities
    
    if (spec.data.empty()) {
        prediction.predictedClass = CnnPredictionClass::UNKNOWN;
        prediction.confidence = 0.0f;
        prediction.isValid = false;
        return prediction;
    }
    
    // Compute simple statistics
    float sum = 0.0f, sumSq = 0.0f;
    float maxVal = 0.0f;
    int maxY = 0;
    
    for (int y = 0; y < spec.height; ++y) {
        for (int x = 0; x < spec.width; ++x) {
            float v = spec.at(x, y);
            sum += v;
            sumSq += v * v;
            
            if (v > maxVal) {
                maxVal = v;
                maxY = y;
            }
        }
    }
    
    float mean = sum / spec.data.size();
    float variance = sumSq / spec.data.size() - mean * mean;
    
    // Heuristics for classification
    float normalScore = 0.6f - variance * 2.0f;  // Low variance -> normal
    float crackleScore = 0.3f;
    float wheezeScore = 0.3f;
    float bothScore = 0.1f;
    
    // High freq energy (top of spectrogram) -> crackle
    float relativeY = static_cast<float>(maxY) / spec.height;
    if (relativeY > 0.5f) {
        crackleScore += 0.3f;
    }
    
    // Mid freq energy -> wheeze
    if (relativeY > 0.3f && relativeY < 0.7f) {
        wheezeScore += 0.3f;
    }
    
    // High variance -> abnormal
    if (variance > 0.1f) {
        normalScore -= 0.3f;
        bothScore += 0.2f;
    }
    
    // Normalize to probabilities
    normalScore = std::max(0.0f, normalScore);
    float total = normalScore + crackleScore + wheezeScore + bothScore;
    
    prediction.probabilities[0] = normalScore / total;
    prediction.probabilities[1] = crackleScore / total;
    prediction.probabilities[2] = wheezeScore / total;
    prediction.probabilities[3] = bothScore / total;
    
    // Find max
    auto maxIt = std::max_element(prediction.probabilities.begin(), 
                                   prediction.probabilities.end());
    int maxIdx = std::distance(prediction.probabilities.begin(), maxIt);
    
    prediction.predictedClass = static_cast<CnnPredictionClass>(maxIdx);
    prediction.confidence = *maxIt;
    prediction.isValid = true;
    
    // Add some randomness for simulation
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-0.05f, 0.05f);
    prediction.confidence = std::clamp(prediction.confidence + noise(rng), 0.0f, 1.0f);
    
    return prediction;
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

CnnInference createMobileNetV2Inference(const std::string& modelPath) {
    CnnConfig config;
    config.modelType = CnnModelType::MOBILENET_V2;
    config.modelPath = modelPath;
    config.inputWidth = 224;
    config.inputHeight = 224;
    config.inputChannels = 1;
    config.numClasses = CNN_NUM_CLASSES;
    
    return CnnInference(config);
}

CnnInference createDpuInference(const std::string& xmodelPath) {
    /**
     * Factory function for Vitis-AI DPU inference
     * 
     * Returns CnnInference with placeholder callback.
     * User needs to implement actual Vitis-AI callback.
     * 
     * Example Vitis-AI integration:
     * 
     *   auto inference = createDpuInference("model.xmodel");
     *   inference.setCustomInferenceCallback([](const std::vector<float>& input) {
     *       // Vitis-AI DPU inference code here
     *       auto runner = vart::Runner::create_runner(xmodel);
     *       // ... process input and get output
     *       return output;
     *   });
     */
    
    CnnConfig config;
    config.modelType = CnnModelType::CUSTOM;
    config.modelPath = xmodelPath;
    config.provider = ExecutionProvider::DPU;
    
    CnnInference inference(config);
    
    // Set placeholder callback with warning
    inference.setCustomInferenceCallback([](const std::vector<float>& input) {
        std::cerr << "[Warning] DPU callback not implemented. "
                  << "Please set actual Vitis-AI callback.\n";
        
        // Return dummy output
        std::vector<float> output(CNN_NUM_CLASSES, 0.25f);
        return output;
    });
    
    return inference;
}

bool validateOnnxModel(const std::string& modelPath) {
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[validateOnnxModel] Cannot open file: " << modelPath << "\n";
        return false;
    }
    
    // Check ONNX magic number (first 4 bytes should be specific pattern)
    char header[4];
    file.read(header, 4);
    
    // ONNX files typically start with protobuf header
    // This is a simplified check
    if (!file.good()) {
        return false;
    }
    
    // Check file size (should be > 1KB for a valid model)
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    
    if (fileSize < 1024) {
        std::cerr << "[validateOnnxModel] File too small to be a valid model\n";
        return false;
    }
    
    return true;
}

std::vector<ExecutionProvider> getSupportedProviders() {
    std::vector<ExecutionProvider> providers;
    
    // CPU is always available
    providers.push_back(ExecutionProvider::CPU);
    
#ifdef USE_ONNX_RUNTIME
    // Check for CUDA
    auto availableProviders = Ort::GetAvailableProviders();
    for (const auto& provider : availableProviders) {
        if (provider == "CUDAExecutionProvider") {
            providers.push_back(ExecutionProvider::CUDA);
        }
        else if (provider == "TensorrtExecutionProvider") {
            providers.push_back(ExecutionProvider::TENSORRT);
        }
        else if (provider == "OpenVINOExecutionProvider") {
            providers.push_back(ExecutionProvider::OPENVINO);
        }
    }
#endif
    
    return providers;
}

} // namespace respiratory
