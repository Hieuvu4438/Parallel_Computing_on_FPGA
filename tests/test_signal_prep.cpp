/**
 * @file test_signal_prep.cpp
 * @brief Unit tests for SignalProcessor, DatasetManager, and FeatureExtractor
 * 
 * Kiểm tra các chức năng của:
 * - Module tiền xử lý tín hiệu (SignalProcessor)
 * - Module quản lý dataset ICBHI (DatasetManager)
 * - Module trích xuất đặc trưng (FeatureExtractor)
 */

#include "SignalPrep.hpp"
#include "FeatureExtraction.h"
#include "CascadedLogic.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;
using namespace respiratory;

// ============================================================================
// TEST HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Tạo tín hiệu sine để test
 */
std::vector<float> generateSineWave(float frequency, float sampleRate, 
                                     float duration, float amplitude = 1.0f) {
    size_t numSamples = static_cast<size_t>(sampleRate * duration);
    std::vector<float> signal(numSamples);
    
    for (size_t i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / sampleRate;
        signal[i] = amplitude * std::sin(2.0f * M_PI * frequency * t);
    }
    
    return signal;
}

/**
 * @brief Tạo tín hiệu nhiễu Gaussian
 */
std::vector<float> generateNoise(size_t numSamples, float amplitude = 0.1f) {
    std::vector<float> noise(numSamples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, amplitude);
    
    for (size_t i = 0; i < numSamples; ++i) {
        noise[i] = dist(gen);
    }
    
    return noise;
}

/**
 * @brief Tính RMS của tín hiệu
 */
float computeRMS(const std::vector<float>& signal) {
    if (signal.empty()) return 0.0f;
    
    float sumSquared = 0.0f;
    for (const auto& s : signal) {
        sumSquared += s * s;
    }
    return std::sqrt(sumSquared / signal.size());
}

/**
 * @brief Kiểm tra xem giá trị có nằm trong khoảng không
 */
bool isInRange(float value, float min, float max) {
    return value >= min && value <= max;
}

// ============================================================================
// TEST CASES
// ============================================================================

/**
 * Test 1: Kiểm tra resampling
 */
bool testResampling() {
    std::cout << "\n[TEST] Resampling..." << std::endl;
    
    SignalProcessor processor;
    
    // Tạo tín hiệu test tại 8kHz
    float inputRate = 8000.0f;
    float targetRate = 4000.0f;
    float frequency = 100.0f;  // 100Hz sine wave
    float duration = 1.0f;
    
    auto inputSignal = generateSineWave(frequency, inputRate, duration);
    std::cout << "  Input samples: " << inputSignal.size() << std::endl;
    
    // Resampling
    std::vector<float> outputSignal;
    processor.resample(inputSignal, inputRate, outputSignal, targetRate);
    
    std::cout << "  Output samples: " << outputSignal.size() << std::endl;
    
    // Kiểm tra số samples đầu ra
    size_t expectedSamples = static_cast<size_t>(inputSignal.size() / 2.0f);
    bool sizeCorrect = std::abs(static_cast<int>(outputSignal.size()) - 
                                static_cast<int>(expectedSamples)) <= 1;
    
    std::cout << "  Expected samples: ~" << expectedSamples << std::endl;
    std::cout << "  Size check: " << (sizeCorrect ? "PASS" : "FAIL") << std::endl;
    
    // Kiểm tra biên độ (nên giữ nguyên ~1.0)
    float maxAbs = findMaxAbsValue(outputSignal);
    bool amplitudeCorrect = isInRange(maxAbs, 0.9f, 1.1f);
    
    std::cout << "  Max amplitude: " << maxAbs << std::endl;
    std::cout << "  Amplitude check: " << (amplitudeCorrect ? "PASS" : "FAIL") << std::endl;
    
    return sizeCorrect && amplitudeCorrect;
}

/**
 * Test 2: Kiểm tra normalization
 */
bool testNormalization() {
    std::cout << "\n[TEST] Normalization..." << std::endl;
    
    SignalProcessor processor;
    
    // Tạo tín hiệu với biên độ khác nhau
    std::vector<float> signal = {0.5f, -0.3f, 0.8f, -0.2f, 0.1f};
    
    std::cout << "  Before normalization: max = " << findMaxAbsValue(signal) << std::endl;
    
    // Normalize
    processor.normalize(signal);
    
    float maxAfter = findMaxAbsValue(signal);
    std::cout << "  After normalization: max = " << maxAfter << std::endl;
    
    // Kiểm tra max = 1.0
    bool normalized = std::abs(maxAfter - 1.0f) < 0.001f;
    
    // Kiểm tra tất cả giá trị trong [-1, 1]
    bool inRange = true;
    for (const auto& s : signal) {
        if (s < -1.0f || s > 1.0f) {
            inRange = false;
            break;
        }
    }
    
    std::cout << "  Normalization check: " << (normalized ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Range check: " << (inRange ? "PASS" : "FAIL") << std::endl;
    
    return normalized && inRange;
}

/**
 * Test 3: Kiểm tra bandpass filter
 */
bool testBandpassFilter() {
    std::cout << "\n[TEST] Bandpass Filter..." << std::endl;
    
    SignalProcessor processor;
    
    float sampleRate = 4000.0f;
    float duration = 1.0f;
    
    // Tạo tín hiệu với nhiều thành phần tần số
    // 30Hz (ngoài dải - sẽ bị loại)
    // 500Hz (trong dải - sẽ được giữ)
    // 2800Hz (ngoài dải - sẽ bị loại)
    
    auto sig30Hz = generateSineWave(30.0f, sampleRate, duration, 1.0f);
    auto sig500Hz = generateSineWave(500.0f, sampleRate, duration, 1.0f);
    auto sig2800Hz = generateSineWave(1900.0f, sampleRate, duration, 1.0f);
    
    // Tổng hợp tín hiệu
    std::vector<float> mixedSignal(sig30Hz.size());
    for (size_t i = 0; i < sig30Hz.size(); ++i) {
        mixedSignal[i] = sig30Hz[i] + sig500Hz[i] + sig2800Hz[i];
    }
    
    float rmsBeforeInBand = computeRMS(sig500Hz);
    float rmsBefore = computeRMS(mixedSignal);
    
    std::cout << "  RMS before filtering: " << rmsBefore << std::endl;
    std::cout << "  RMS of in-band signal (500Hz): " << rmsBeforeInBand << std::endl;
    
    // Áp dụng bandpass filter
    std::vector<float> filteredSignal;
    processor.applyBandpassFilter(mixedSignal, filteredSignal, sampleRate, 50.0f, 2500.0f);
    
    float rmsAfter = computeRMS(filteredSignal);
    std::cout << "  RMS after filtering: " << rmsAfter << std::endl;
    
    // Tín hiệu sau lọc nên gần với tín hiệu 500Hz (in-band)
    // RMS sau lọc nên nhỏ hơn hoặc gần bằng RMS của tín hiệu in-band
    bool filterWorking = rmsAfter < rmsBefore && rmsAfter > 0.3f;
    
    std::cout << "  Filter effectiveness: " << (filterWorking ? "PASS" : "FAIL") << std::endl;
    
    return filterWorking;
}

/**
 * Test 4: Kiểm tra breathing cycle segmentation
 */
bool testSegmentation() {
    std::cout << "\n[TEST] Breathing Cycle Segmentation..." << std::endl;
    
    SignalProcessor processor;
    
    float sampleRate = 4000.0f;
    
    // Tạo tín hiệu mô phỏng 3 chu kỳ hô hấp
    // Mỗi chu kỳ: 1.5 giây hoạt động + 0.5 giây nghỉ
    
    std::vector<float> signal;
    
    for (int cycle = 0; cycle < 3; ++cycle) {
        // Phần hoạt động (1.5 giây)
        auto activeSignal = generateSineWave(200.0f, sampleRate, 1.5f, 0.8f);
        signal.insert(signal.end(), activeSignal.begin(), activeSignal.end());
        
        // Phần nghỉ (0.5 giây - nhiễu nhỏ)
        auto silentSignal = generateNoise(static_cast<size_t>(sampleRate * 0.5f), 0.02f);
        signal.insert(signal.end(), silentSignal.begin(), silentSignal.end());
    }
    
    std::cout << "  Total samples: " << signal.size() << std::endl;
    std::cout << "  Expected cycles: 3" << std::endl;
    
    // Phân đoạn
    std::vector<BreathingCycle> cycles;
    processor.segmentBreathingCycles(signal, cycles, sampleRate);
    
    std::cout << "  Detected cycles: " << cycles.size() << std::endl;
    
    // In thông tin các chu kỳ
    for (size_t i = 0; i < cycles.size(); ++i) {
        std::cout << "    Cycle " << i << ": start=" << cycles[i].startIndex 
                  << ", duration=" << cycles[i].duration << "s"
                  << ", energy=" << cycles[i].averageEnergy << std::endl;
    }
    
    // Kiểm tra: nên có ít nhất 1 chu kỳ
    bool hasSegments = !cycles.empty();
    std::cout << "  Segmentation check: " << (hasSegments ? "PASS" : "FAIL") << std::endl;
    
    return hasSegments;
}

/**
 * Test 5: Kiểm tra processBuffer (full pipeline)
 */
bool testFullPipeline() {
    std::cout << "\n[TEST] Full Processing Pipeline (Buffer)..." << std::endl;
    
    SignalProcessor processor;
    
    // Tạo tín hiệu test tại 8kHz
    float inputRate = 8000.0f;
    float duration = 3.0f;
    
    // Tín hiệu hô hấp giả lập: 200Hz sine + noise
    auto baseSignal = generateSineWave(200.0f, inputRate, duration, 0.7f);
    auto noise = generateNoise(baseSignal.size(), 0.1f);
    
    // Thêm một số "burst" để mô phỏng chu kỳ hô hấp
    for (size_t i = 0; i < baseSignal.size(); ++i) {
        baseSignal[i] += noise[i];
        
        // Tạo envelope giả lập hô hấp
        float t = static_cast<float>(i) / inputRate;
        float envelope = 0.5f + 0.5f * std::sin(2.0f * M_PI * 0.3f * t);  // ~0.3Hz breathing
        baseSignal[i] *= envelope;
    }
    
    std::cout << "  Input: " << baseSignal.size() << " samples at " << inputRate << " Hz" << std::endl;
    
    // Xử lý
    std::vector<BreathingCycle> cycles;
    bool success = processor.processBuffer(baseSignal, inputRate, cycles);
    
    std::cout << "  Processing: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    
    if (success) {
        const auto& processed = processor.getProcessedSignal();
        std::cout << "  Output: " << processed.size() << " samples at " 
                  << processor.getCurrentSampleRate() << " Hz" << std::endl;
        std::cout << "  Detected cycles: " << cycles.size() << std::endl;
        
        // Kiểm tra normalized
        float maxAbs = findMaxAbsValue(processed);
        std::cout << "  Max amplitude: " << maxAbs << std::endl;
        
        bool isNormalized = isInRange(maxAbs, 0.99f, 1.01f);
        std::cout << "  Normalized check: " << (isNormalized ? "PASS" : "FAIL") << std::endl;
        
        return isNormalized;
    }
    
    return false;
}

// ============================================================================
// DATASET MANAGER TESTS
// ============================================================================

/**
 * Test 6: Kiểm tra parse filename ICBHI
 */
bool testParseFilename() {
    std::cout << "\n[TEST] Parse ICBHI Filename..." << std::endl;
    
    DatasetManager manager(".");  // Path doesn't matter for this test
    
    // Test case 1: Standard filename
    PatientInfo info1;
    bool result1 = manager.parseFilename("101_1b1_Al_sc_Meditron.wav", info1);
    
    std::cout << "  Filename: 101_1b1_Al_sc_Meditron.wav" << std::endl;
    std::cout << "    PatientID: " << info1.patientId << " (expected: 101)" << std::endl;
    std::cout << "    Location: " << info1.chestLocation << " (expected: Al)" << std::endl;
    std::cout << "    Equipment: " << info1.equipment << " (expected: Meditron)" << std::endl;
    
    bool test1Pass = result1 && 
                     info1.patientId == 101 && 
                     info1.chestLocation == "Al" &&
                     info1.equipment == "Meditron";
    
    // Test case 2: Another filename format
    PatientInfo info2;
    bool result2 = manager.parseFilename("226_2b3_Pr_mc_AKGC417L.wav", info2);
    
    bool test2Pass = result2 && info2.patientId == 226;
    
    bool allPassed = test1Pass && test2Pass;
    std::cout << "  Parse check: " << (allPassed ? "PASS" : "FAIL") << std::endl;
    
    return allPassed;
}

/**
 * Test 7: Kiểm tra annotation file parsing
 */
bool testReadAnnotation() {
    std::cout << "\n[TEST] Read Annotation File..." << std::endl;
    
    // Tạo file annotation tạm
    std::string tempFile = "/tmp/test_annotation.txt";
    {
        std::ofstream ofs(tempFile);
        ofs << "0.036\t0.579\t0\t0\n";
        ofs << "0.579\t2.45\t1\t0\n";
        ofs << "2.45\t3.893\t0\t1\n";
        ofs << "3.893\t5.793\t1\t1\n";
    }
    
    DatasetManager manager(".");
    std::vector<CycleAnnotation> annotations;
    
    bool readSuccess = manager.readAnnotationFile(tempFile, annotations);
    
    std::cout << "  Read success: " << (readSuccess ? "YES" : "NO") << std::endl;
    std::cout << "  Annotations count: " << annotations.size() << " (expected: 4)" << std::endl;
    
    bool countCorrect = annotations.size() == 4;
    
    // Check labels
    bool labelsCorrect = true;
    if (countCorrect) {
        labelsCorrect = 
            (annotations[0].getLabel() == RespiratoryLabel::NORMAL) &&
            (annotations[1].getLabel() == RespiratoryLabel::CRACKLE) &&
            (annotations[2].getLabel() == RespiratoryLabel::WHEEZE) &&
            (annotations[3].getLabel() == RespiratoryLabel::BOTH);
        
        std::cout << "  Labels check:" << std::endl;
        std::cout << "    [0] Normal: " << (annotations[0].getLabel() == RespiratoryLabel::NORMAL ? "PASS" : "FAIL") << std::endl;
        std::cout << "    [1] Crackle: " << (annotations[1].getLabel() == RespiratoryLabel::CRACKLE ? "PASS" : "FAIL") << std::endl;
        std::cout << "    [2] Wheeze: " << (annotations[2].getLabel() == RespiratoryLabel::WHEEZE ? "PASS" : "FAIL") << std::endl;
        std::cout << "    [3] Both: " << (annotations[3].getLabel() == RespiratoryLabel::BOTH ? "PASS" : "FAIL") << std::endl;
    }
    
    // Cleanup
    fs::remove(tempFile);
    
    bool allPassed = readSuccess && countCorrect && labelsCorrect;
    std::cout << "  Annotation check: " << (allPassed ? "PASS" : "FAIL") << std::endl;
    
    return allPassed;
}

/**
 * Test 8: Kiểm tra segmentation by annotations
 */
bool testSegmentByAnnotations() {
    std::cout << "\n[TEST] Segment By Annotations..." << std::endl;
    
    // Tạo tín hiệu test (3 giây @ 4kHz = 12000 samples)
    std::vector<float> signal(12000);
    for (size_t i = 0; i < signal.size(); ++i) {
        signal[i] = std::sin(2.0f * M_PI * 100.0f * i / 4000.0f);
    }
    
    // Tạo annotations (2 chu kỳ)
    std::vector<CycleAnnotation> annotations(2);
    annotations[0].startTime = 0.0f;
    annotations[0].endTime = 1.5f;
    annotations[0].hasCrackle = true;
    annotations[0].hasWheeze = false;
    
    annotations[1].startTime = 1.5f;
    annotations[1].endTime = 3.0f;
    annotations[1].hasCrackle = false;
    annotations[1].hasWheeze = true;
    
    // Thông tin patient giả
    PatientInfo patientInfo;
    patientInfo.patientId = 999;
    
    DatasetManager manager(".");
    std::vector<LabeledBreathingCycle> cycles;
    
    manager.segmentByAnnotations(signal, 4000.0f, annotations, 
                                  patientInfo, "test.wav", cycles);
    
    std::cout << "  Segments created: " << cycles.size() << " (expected: 2)" << std::endl;
    
    bool countCorrect = cycles.size() == 2;
    bool labelsCorrect = true;
    bool samplesCorrect = true;
    
    if (countCorrect) {
        // Kiểm tra nhãn
        labelsCorrect = (cycles[0].label == RespiratoryLabel::CRACKLE) &&
                        (cycles[1].label == RespiratoryLabel::WHEEZE);
        
        // Kiểm tra số samples (1.5s @ 4kHz = 6000 samples)
        samplesCorrect = (cycles[0].samples.size() == 6000) &&
                         (cycles[1].samples.size() == 6000);
        
        std::cout << "  Cycle 0: " << cycles[0].samples.size() << " samples, label=" 
                  << cycles[0].getLabelString() << std::endl;
        std::cout << "  Cycle 1: " << cycles[1].samples.size() << " samples, label=" 
                  << cycles[1].getLabelString() << std::endl;
    }
    
    bool allPassed = countCorrect && labelsCorrect && samplesCorrect;
    std::cout << "  Segment by annotations: " << (allPassed ? "PASS" : "FAIL") << std::endl;
    
    return allPassed;
}

/**
 * Test 9: Kiểm tra dataset split
 */
bool testDatasetSplit() {
    std::cout << "\n[TEST] Dataset Split..." << std::endl;
    
    // Tạo 100 cycles giả
    std::vector<LabeledBreathingCycle> allCycles(100);
    for (int i = 0; i < 100; ++i) {
        allCycles[i].patientInfo.patientId = i % 10;
        allCycles[i].label = static_cast<RespiratoryLabel>(i % 4);
    }
    
    std::vector<LabeledBreathingCycle> trainSet, testSet;
    
    // Split 80/20
    DatasetManager::splitDataset(allCycles, 0.8f, trainSet, testSet, 42);
    
    std::cout << "  Total: " << allCycles.size() << std::endl;
    std::cout << "  Train: " << trainSet.size() << " (expected: ~80)" << std::endl;
    std::cout << "  Test: " << testSet.size() << " (expected: ~20)" << std::endl;
    
    bool splitCorrect = (trainSet.size() == 80) && (testSet.size() == 20);
    std::cout << "  Split check: " << (splitCorrect ? "PASS" : "FAIL") << std::endl;
    
    return splitCorrect;
}

// ============================================================================
// FEATURE EXTRACTOR TESTS
// ============================================================================

/**
 * Test 10: Kiểm tra ZCR calculation
 */
bool testZCR() {
    std::cout << "\n[TEST] Zero Crossing Rate..." << std::endl;
    
    FeatureExtractor extractor;
    
    // Test với tín hiệu sine - có nhiều zero crossings
    std::vector<float> sineWave(1000);
    for (int i = 0; i < 1000; ++i) {
        sineWave[i] = std::sin(2.0f * M_PI * 100.0f * i / 4000.0f);
    }
    
    float zcr = extractor.computeZCR(sineWave);
    std::cout << "  Sine wave ZCR: " << zcr << std::endl;
    
    // ZCR của sine wave 100Hz @ 4kHz nên khoảng 0.05 (200 crossings / 4000 samples)
    bool sinePass = (zcr > 0.04f && zcr < 0.06f);
    
    // Test với tín hiệu DC - không có zero crossings
    std::vector<float> dcSignal(1000, 0.5f);
    float zcrDc = extractor.computeZCR(dcSignal);
    std::cout << "  DC signal ZCR: " << zcrDc << std::endl;
    bool dcPass = (zcrDc < 0.001f);
    
    std::cout << "  ZCR test: " << ((sinePass && dcPass) ? "PASS" : "FAIL") << std::endl;
    
    return sinePass && dcPass;
}

/**
 * Test 11: Kiểm tra RMSE calculation
 */
bool testRMSE() {
    std::cout << "\n[TEST] Root Mean Square Energy..." << std::endl;
    
    FeatureExtractor extractor;
    
    // Test với tín hiệu có amplitude 1
    std::vector<float> signal(1000, 1.0f);
    float rmse1 = extractor.computeRMSE(signal);
    std::cout << "  Constant 1.0 RMSE: " << rmse1 << " (expected: 1.0)" << std::endl;
    bool test1 = std::abs(rmse1 - 1.0f) < 0.001f;
    
    // Test với tín hiệu sine (amplitude 1) - RMSE = 1/sqrt(2) ≈ 0.707
    std::vector<float> sineWave(1000);
    for (int i = 0; i < 1000; ++i) {
        sineWave[i] = std::sin(2.0f * M_PI * 50.0f * i / 1000.0f);
    }
    float rmseSine = extractor.computeRMSE(sineWave);
    std::cout << "  Sine wave RMSE: " << rmseSine << " (expected: ~0.707)" << std::endl;
    bool test2 = std::abs(rmseSine - 0.707f) < 0.01f;
    
    std::cout << "  RMSE test: " << ((test1 && test2) ? "PASS" : "FAIL") << std::endl;
    
    return test1 && test2;
}

/**
 * Test 12: Kiểm tra Hz to Mel conversion
 */
bool testMelConversion() {
    std::cout << "\n[TEST] Mel Scale Conversion..." << std::endl;
    
    // Test Hz to Mel
    float mel1000 = FeatureExtractor::hzToMel(1000.0f);
    std::cout << "  1000 Hz -> " << mel1000 << " Mel (expected: ~1000)" << std::endl;
    bool test1 = std::abs(mel1000 - 1000.0f) < 10.0f;  // Approximately 1000 Mel at 1000 Hz
    
    // Test round-trip conversion
    float hz = 500.0f;
    float mel = FeatureExtractor::hzToMel(hz);
    float hzBack = FeatureExtractor::melToHz(mel);
    std::cout << "  Round-trip: " << hz << " Hz -> " << mel << " Mel -> " << hzBack << " Hz" << std::endl;
    bool test2 = std::abs(hz - hzBack) < 0.1f;
    
    std::cout << "  Mel conversion test: " << ((test1 && test2) ? "PASS" : "FAIL") << std::endl;
    
    return test1 && test2;
}

/**
 * Test 13: Kiểm tra MFCC extraction
 */
bool testMFCCExtraction() {
    std::cout << "\n[TEST] MFCC Extraction..." << std::endl;
    
    FeatureExtractor extractor(4000.0f);
    
    // Tạo tín hiệu test (2 giây @ 4kHz = 8000 samples)
    std::vector<float> signal(8000);
    for (int i = 0; i < 8000; ++i) {
        // Tín hiệu với nhiều thành phần tần số
        signal[i] = 0.5f * std::sin(2.0f * M_PI * 200.0f * i / 4000.0f) +
                    0.3f * std::sin(2.0f * M_PI * 500.0f * i / 4000.0f) +
                    0.2f * std::sin(2.0f * M_PI * 1000.0f * i / 4000.0f);
    }
    
    CycleFeatures features;
    bool success = extractor.extractFeatures(signal, features, false);
    
    std::cout << "  Extraction success: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "  Number of frames: " << features.numFrames << std::endl;
    std::cout << "  Duration: " << features.durationSec << "s" << std::endl;
    
    bool hasFrames = features.numFrames > 0;
    bool hasMFCC = features.mfcc_mean.size() == NUM_MFCC_COEFFS;
    bool hasDelta = features.delta_mean.size() == NUM_MFCC_COEFFS;
    
    std::cout << "  MFCC coefficients: " << features.mfcc_mean.size() << " (expected: 13)" << std::endl;
    std::cout << "  First 3 MFCC: [" << features.mfcc_mean[0] << ", " 
              << features.mfcc_mean[1] << ", " << features.mfcc_mean[2] << "]" << std::endl;
    
    bool allPassed = success && hasFrames && hasMFCC && hasDelta;
    std::cout << "  MFCC extraction test: " << (allPassed ? "PASS" : "FAIL") << std::endl;
    
    return allPassed;
}

/**
 * Test 14: Kiểm tra flat feature vector
 */
bool testFlatFeatureVector() {
    std::cout << "\n[TEST] Flat Feature Vector..." << std::endl;
    
    FeatureExtractor extractor(4000.0f);
    
    // Tạo tín hiệu test
    std::vector<float> signal(4000);  // 1 second
    for (int i = 0; i < 4000; ++i) {
        signal[i] = std::sin(2.0f * M_PI * 100.0f * i / 4000.0f);
    }
    
    std::vector<float> flatFeatures;
    bool success = extractor.extractFlatFeatures(signal, flatFeatures);
    
    int expectedDim = CycleFeatures::getFeatureDimension();
    
    std::cout << "  Extraction success: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "  Flat vector size: " << flatFeatures.size() << " (expected: " << expectedDim << ")" << std::endl;
    
    bool sizeCorrect = (flatFeatures.size() == static_cast<size_t>(expectedDim));
    
    // Kiểm tra không có NaN hoặc Inf
    bool noNaN = true;
    for (float val : flatFeatures) {
        if (std::isnan(val) || std::isinf(val)) {
            noNaN = false;
            break;
        }
    }
    std::cout << "  No NaN/Inf values: " << (noNaN ? "YES" : "NO") << std::endl;
    
    bool allPassed = success && sizeCorrect && noNaN;
    std::cout << "  Flat vector test: " << (allPassed ? "PASS" : "FAIL") << std::endl;
    
    return allPassed;
}

// ============================================================================
// CASCADED CONTROLLER TESTS
// ============================================================================

/**
 * @brief Test SeptupleForest voting mechanism
 */
bool testSeptupleForestVoting() {
    std::cout << "\n[Test] Septuple Forest Voting" << std::endl;
    
    // Tạo training data giả
    std::vector<std::vector<float>> features;
    std::vector<int> labels;
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Generate 100 samples, 6 features each
    for (int i = 0; i < 100; ++i) {
        std::vector<float> sample(6);
        for (int j = 0; j < 6; ++j) {
            sample[j] = dist(rng);
        }
        features.push_back(sample);
        labels.push_back(i % 4);  // 4 classes
    }
    
    // Train Septuple Forest
    SeptupleForest forest(5, 4);  // 5 trees per cluster, consensus threshold 4
    forest.train(features, labels);
    
    // Check if trained
    bool trained = forest.isTrained();
    std::cout << "  Forest trained: " << (trained ? "YES" : "NO") << std::endl;
    
    // Test prediction
    std::vector<float> testSample = {0.5f, -0.3f, 0.8f, -0.1f, 0.2f, 0.6f};
    LayerResult result;
    bool hasConsensus = forest.predictWithVoting(testSample, result);
    
    std::cout << "  Has consensus: " << (hasConsensus ? "YES" : "NO") << std::endl;
    std::cout << "  Vote counts: [";
    for (int i = 0; i < NUM_CLASSES; ++i) {
        std::cout << result.voteCounts[i];
        if (i < NUM_CLASSES - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Predicted class: " << predictionClassToString(result.consensusClass) << std::endl;
    std::cout << "  Confidence: " << result.aggregatedConfidence << std::endl;
    
    bool success = trained && result.consensusClass != PredictionClass::UNKNOWN;
    std::cout << "  Septuple Forest voting test: " << (success ? "PASS" : "FAIL") << std::endl;
    
    return success;
}

/**
 * @brief Test CascadedController state machine
 */
bool testCascadedStateMachine() {
    std::cout << "\n[Test] Cascaded Controller State Machine" << std::endl;
    
    // Create controller
    CascadedController controller;
    
    // Check initial state (not trained)
    bool notTrained = !controller.isFullyTrained();
    std::cout << "  Initial state (not trained): " << (notTrained ? "YES" : "NO") << std::endl;
    
    // Generate training data
    std::vector<FeatureVector> trainFeatures;
    std::vector<int> trainLabels;
    
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < 50; ++i) {
        FeatureVector fv;
        fv.globalEnergy = std::abs(dist(rng));
        fv.energyVariance = std::abs(dist(rng));
        fv.rmse_mean = std::abs(dist(rng));
        fv.rmse_std = std::abs(dist(rng));
        fv.eed = std::abs(dist(rng));
        fv.zcr_mean = std::abs(dist(rng));
        fv.zcr_std = std::abs(dist(rng));
        fv.amplitudeSkewness = dist(rng);
        fv.amplitudeKurtosis = 3.0f + dist(rng);
        
        for (int j = 0; j < 13; ++j) {
            fv.mfcc_mean[j] = dist(rng);
            fv.mfcc_std[j] = std::abs(dist(rng));
            fv.delta_mean[j] = dist(rng);
            fv.delta_std[j] = std::abs(dist(rng));
            fv.delta2_mean[j] = dist(rng);
            fv.delta2_std[j] = std::abs(dist(rng));
        }
        
        trainFeatures.push_back(fv);
        trainLabels.push_back(i % 4);
    }
    
    // Train
    controller.trainAll(trainFeatures, trainLabels);
    
    bool trained = controller.isFullyTrained();
    std::cout << "  After training (fully trained): " << (trained ? "YES" : "NO") << std::endl;
    
    // Test prediction
    PredictionResult result = controller.predict(trainFeatures[0]);
    
    std::cout << "  Prediction result:" << std::endl;
    std::cout << "    Class: " << result.getClassName() << std::endl;
    std::cout << "    Confidence: " << result.confidence << std::endl;
    std::cout << "    Exit layer: " << cascadeLayerToString(result.exitLayer) << std::endl;
    std::cout << "    Exit reason: " << exitReasonToString(result.exitReason) << std::endl;
    std::cout << "    Layers processed: " << result.totalLayersProcessed << std::endl;
    
    // Check statistics
    auto stats = controller.getExitStatistics();
    std::cout << "  Exit statistics: L1=" << stats.layer1Exits 
              << ", L2=" << stats.layer2Exits
              << ", L3=" << stats.layer3Exits << std::endl;
    
    bool validPrediction = result.predictedClass != PredictionClass::UNKNOWN;
    bool hasLayers = result.totalLayersProcessed > 0;
    
    bool success = notTrained && trained && validPrediction && hasLayers;
    std::cout << "  State machine test: " << (success ? "PASS" : "FAIL") << std::endl;
    
    return success;
}

/**
 * @brief Test FeatureVector layer extraction
 */
bool testFeatureVectorLayers() {
    std::cout << "\n[Test] FeatureVector Layer Extraction" << std::endl;
    
    FeatureVector fv;
    fv.globalEnergy = 0.5f;
    fv.energyVariance = 0.1f;
    fv.rmse_mean = 0.3f;
    fv.rmse_std = 0.05f;
    fv.eed = 0.2f;
    fv.zcr_mean = 0.15f;
    fv.zcr_std = 0.02f;
    fv.amplitudeSkewness = -0.5f;
    fv.amplitudeKurtosis = 4.0f;
    fv.metadata.hasPriorCondition = true;
    
    for (int i = 0; i < 13; ++i) {
        fv.mfcc_mean[i] = static_cast<float>(i);
        fv.mfcc_std[i] = static_cast<float>(i) * 0.1f;
        fv.delta_mean[i] = static_cast<float>(i) * 0.5f;
        fv.delta_std[i] = static_cast<float>(i) * 0.05f;
        fv.delta2_mean[i] = static_cast<float>(i) * 0.25f;
        fv.delta2_std[i] = static_cast<float>(i) * 0.025f;
    }
    
    auto layer1 = fv.getLayer1Features();
    auto layer2 = fv.getLayer2Features();
    auto layer3 = fv.getLayer3Features();
    auto flat = fv.toFlatVector();
    
    std::cout << "  Layer 1 features size: " << layer1.size() << " (expected: 6)" << std::endl;
    std::cout << "  Layer 2 features size: " << layer2.size() << " (expected: 7)" << std::endl;
    std::cout << "  Layer 3 features size: " << layer3.size() << " (expected: 78)" << std::endl;
    std::cout << "  Flat vector size: " << flat.size() << std::endl;
    
    bool l1Size = (layer1.size() == 6);
    bool l2Size = (layer2.size() == 7);
    bool l3Size = (layer3.size() == 78);
    
    // Check Layer 1 values
    bool l1Values = (std::abs(layer1[0] - 0.5f) < 1e-5f) &&   // globalEnergy
                    (std::abs(layer1[1] - 0.1f) < 1e-5f) &&   // energyVariance
                    (std::abs(layer1[5] - 1.0f) < 1e-5f);     // hasPriorCondition
    
    // Check Layer 2 values
    bool l2Values = (std::abs(layer2[0] - 0.15f) < 1e-5f) &&  // zcr_mean
                    (std::abs(layer2[1] - 0.02f) < 1e-5f);    // zcr_std
    
    bool success = l1Size && l2Size && l3Size && l1Values && l2Values;
    std::cout << "  Feature vector layers test: " << (success ? "PASS" : "FAIL") << std::endl;
    
    return success;
}

/**
 * @brief Test prediction class conversion
 */
bool testPredictionClassConversion() {
    std::cout << "\n[Test] Prediction Class Conversion" << std::endl;
    
    bool normal = (predictionClassToString(PredictionClass::NORMAL) == "Normal");
    bool crackle = (predictionClassToString(PredictionClass::CRACKLE) == "Crackle");
    bool wheeze = (predictionClassToString(PredictionClass::WHEEZE) == "Wheeze");
    bool both = (predictionClassToString(PredictionClass::BOTH) == "Both");
    bool unknown = (predictionClassToString(PredictionClass::UNKNOWN) == "Unknown");
    bool needCnn = (predictionClassToString(PredictionClass::NEED_CNN) == "Need_CNN");
    
    std::cout << "  Normal: " << (normal ? "OK" : "FAIL") << std::endl;
    std::cout << "  Crackle: " << (crackle ? "OK" : "FAIL") << std::endl;
    std::cout << "  Wheeze: " << (wheeze ? "OK" : "FAIL") << std::endl;
    std::cout << "  Both: " << (both ? "OK" : "FAIL") << std::endl;
    std::cout << "  Unknown: " << (unknown ? "OK" : "FAIL") << std::endl;
    std::cout << "  Need_CNN: " << (needCnn ? "OK" : "FAIL") << std::endl;
    
    bool success = normal && crackle && wheeze && both && unknown && needCnn;
    std::cout << "  Prediction class conversion test: " << (success ? "PASS" : "FAIL") << std::endl;
    
    return success;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Complete System Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int total = 18;
    
    // SignalProcessor tests
    std::cout << "\n--- SignalProcessor Tests ---" << std::endl;
    if (testResampling()) passed++;
    if (testNormalization()) passed++;
    if (testBandpassFilter()) passed++;
    if (testSegmentation()) passed++;
    if (testFullPipeline()) passed++;
    
    // DatasetManager tests
    std::cout << "\n--- DatasetManager Tests ---" << std::endl;
    if (testParseFilename()) passed++;
    if (testReadAnnotation()) passed++;
    if (testSegmentByAnnotations()) passed++;
    if (testDatasetSplit()) passed++;
    
    // FeatureExtractor tests
    std::cout << "\n--- FeatureExtractor Tests ---" << std::endl;
    if (testZCR()) passed++;
    if (testRMSE()) passed++;
    if (testMelConversion()) passed++;
    if (testMFCCExtraction()) passed++;
    if (testFlatFeatureVector()) passed++;
    
    // CascadedController tests
    std::cout << "\n--- CascadedController Tests ---" << std::endl;
    if (testSeptupleForestVoting()) passed++;
    if (testCascadedStateMachine()) passed++;
    if (testFeatureVectorLayers()) passed++;
    if (testPredictionClassConversion()) passed++;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (passed == total) ? 0 : 1;
}
