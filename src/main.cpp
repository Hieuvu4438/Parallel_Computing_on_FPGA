/**
 * @file main.cpp
 * @brief Main entry point for Respiratory Sound Analysis System
 * 
 * Chương trình chính demo pipeline phân tích âm thanh hô hấp
 * theo Cascaded Framework với ICBHI 2017 Dataset.
 * 
 * Pipeline:
 *   Phase 1: Signal Preprocessing (Resample, Filter, Normalize, Segment)
 *   Phase 2: Feature Extraction (EED, ZCR, RMSE, MFCC)
 *   Phase 3: Classification (Random Forest / CNN) [Future]
 * 
 * Usage:
 *   ./respiratory_analysis                    # Load toàn bộ dataset
 *   ./respiratory_analysis <file.wav>         # Xử lý một file
 *   ./respiratory_analysis --test             # Chạy với 10 file đầu tiên
 *   ./respiratory_analysis --features         # Test feature extraction
 */

#include "SignalPrep.hpp"
#include "FeatureExtraction.h"
#include "CascadedLogic.h"
#include "Common.h"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <set>
#include <fstream>
#include <random>

namespace fs = std::filesystem;
using namespace respiratory;

void printUsage(const char* programName) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Respiratory Sound Analysis - Cascaded Framework          ║\n";
    std::cout << "║     Based on IEEE Paper for FPGA Implementation              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Usage: " << programName << " [options] [input.wav]\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --help, -h       Show this help message\n";
    std::cout << "  --test           Quick test with first 10 files\n";
    std::cout << "  --features       Test feature extraction on sample data\n";
    std::cout << "  --extract        Extract features and save to CSV\n";
    std::cout << "  --classify       Train & test Cascaded Classifier\n";
    std::cout << "  --stats          Load dataset and show statistics only\n";
    std::cout << "  --path <dir>     Specify custom dataset path\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << "                              # Process full dataset\n";
    std::cout << "  " << programName << " --test                       # Quick test\n";
    std::cout << "  " << programName << " --features                   # Test feature extraction\n";
    std::cout << "  " << programName << " --extract                    # Extract & save features\n";
    std::cout << "  " << programName << " --classify                   # Train & evaluate classifier\n";
    std::cout << "  " << programName << " --classify --test            # Quick classifier test\n";
    std::cout << "\n";
    std::cout << "Processing Pipeline:\n";
    std::cout << "  Phase 1: Signal Preprocessing\n";
    std::cout << "    - Resample to 4kHz\n";
    std::cout << "    - Bandpass filter (50-2500 Hz)\n";
    std::cout << "    - Normalize to [-1, 1]\n";
    std::cout << "    - Segment by ICBHI annotations\n";
    std::cout << "  Phase 2: Feature Extraction\n";
    std::cout << "    - Time-domain: EED, ZCR, RMSE\n";
    std::cout << "    - Frequency: MFCC (39-dim)\n";
    std::cout << "  Phase 3: Cascaded Classification\n";
    std::cout << "    - Layer 1: Global Screening (RF)\n";
    std::cout << "    - Layer 2: Transient Detection (RF)\n";
    std::cout << "    - Layer 3: Spectral Screening (RF)\n";
    std::cout << "    - Layer 4: CNN (for ambiguous cases)\n";
    std::cout << "\n";
}

/**
 * @brief Xử lý một file WAV đơn lẻ
 */
int processSingleFile(const std::string& filePath) {
    std::cout << "\n[Mode] Single File Processing\n";
    std::cout << "Input: " << filePath << "\n\n";
    
    SignalProcessor processor;
    std::vector<BreathingCycle> cycles;
    
    if (!processor.processFile(filePath, cycles)) {
        std::cerr << "Error: Failed to process file!\n";
        return 1;
    }
    
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════╗\n";
    std::cout << "║         Processing Results            ║\n";
    std::cout << "╠═══════════════════════════════════════╣\n";
    std::cout << "║  Signal length: " << std::setw(8) << processor.getProcessedSignal().size() 
              << " samples    ║\n";
    std::cout << "║  Sample rate:   " << std::setw(8) << processor.getCurrentSampleRate() 
              << " Hz        ║\n";
    std::cout << "║  Detected cycles: " << std::setw(6) << cycles.size() 
              << "              ║\n";
    std::cout << "╚═══════════════════════════════════════╝\n";
    
    // In chi tiết các chu kỳ
    if (!cycles.empty()) {
        std::cout << "\nBreathing Cycles:\n";
        for (size_t i = 0; i < cycles.size() && i < 10; ++i) {
            std::cout << "  [" << (i + 1) << "] duration=" << std::fixed << std::setprecision(2)
                      << cycles[i].duration << "s, samples=" << cycles[i].samples.size()
                      << ", energy=" << std::scientific << std::setprecision(3)
                      << cycles[i].averageEnergy << "\n";
        }
        if (cycles.size() > 10) {
            std::cout << "  ... and " << (cycles.size() - 10) << " more cycles\n";
        }
    }
    
    return 0;
}

/**
 * @brief Load và xử lý toàn bộ ICBHI dataset
 */
int processDataset(const std::string& datasetPath, bool quickTest = false) {
    std::cout << "\n[Mode] ICBHI Dataset Processing\n";
    std::cout << "Path: " << datasetPath << "\n";
    if (quickTest) {
        std::cout << "Note: Quick test mode - processing first 10 files only\n";
    }
    std::cout << "\n";
    
    // Khởi tạo DatasetManager
    DatasetManager manager(datasetPath);
    
    if (!manager.isValidDatasetPath()) {
        std::cerr << "Error: Invalid dataset path or no WAV files found!\n";
        std::cerr << "Please ensure the path contains ICBHI dataset files.\n";
        return 1;
    }
    
    // Đo thời gian
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<LabeledBreathingCycle> dataset;
    
    if (quickTest) {
        // Quick test: chỉ xử lý 10 file đầu
        auto wavFiles = manager.getWavFiles();
        size_t numToProcess = std::min(wavFiles.size(), size_t(10));
        
        std::cout << "Processing " << numToProcess << " files...\n\n";
        
        for (size_t i = 0; i < numToProcess; ++i) {
            std::vector<LabeledBreathingCycle> fileCycles;
            
            std::cout << "  [" << (i + 1) << "/" << numToProcess << "] " 
                      << fs::path(wavFiles[i]).filename().string();
            
            if (manager.processFile(wavFiles[i], fileCycles)) {
                std::cout << " -> " << fileCycles.size() << " cycles\n";
                for (auto& cycle : fileCycles) {
                    dataset.push_back(std::move(cycle));
                }
            } else {
                std::cout << " -> FAILED\n";
            }
        }
    } else {
        // Full dataset processing với progress
        std::cout << "Loading full dataset...\n";
        
        manager.loadDatasetWithProgress(dataset,
            [](size_t current, size_t total) {
                if (current % 50 == 0 || current == total) {
                    float percent = 100.0f * current / total;
                    std::cout << "\r  Progress: " << current << "/" << total 
                              << " (" << std::fixed << std::setprecision(1) 
                              << percent << "%)    " << std::flush;
                }
            });
        std::cout << "\n";
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    // In kết quả
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "                      PROCESSING COMPLETE                       \n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Total cycles loaded: " << dataset.size() << "\n";
    std::cout << "  Processing time:     " << duration.count() / 1000.0 << " seconds\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    // Thống kê chi tiết
    if (!dataset.empty()) {
        // Tính thống kê từ dữ liệu đã load
        DatasetStatistics stats;
        stats.totalCycles = dataset.size();
        std::set<int> uniquePatients;
        float totalDuration = 0.0f;
        
        for (const auto& cycle : dataset) {
            uniquePatients.insert(cycle.patientInfo.patientId);
            totalDuration += cycle.duration;
            
            switch (cycle.label) {
                case RespiratoryLabel::NORMAL: stats.normalCount++; break;
                case RespiratoryLabel::CRACKLE: stats.crackleCount++; break;
                case RespiratoryLabel::WHEEZE: stats.wheezeCount++; break;
                case RespiratoryLabel::BOTH: stats.bothCount++; break;
            }
        }
        
        stats.uniquePatients = uniquePatients.size();
        stats.avgCycleDuration = totalDuration / stats.totalCycles;
        stats.totalFiles = quickTest ? 10 : manager.getStatistics().totalFiles;
        
        stats.print();
        
        // Hiển thị một vài mẫu
        std::cout << "Sample entries:\n";
        for (size_t i = 0; i < std::min(dataset.size(), size_t(5)); ++i) {
            const auto& cycle = dataset[i];
            std::cout << "  [" << (i + 1) << "] Patient " << cycle.patientInfo.patientId
                      << " | " << std::setw(7) << cycle.getLabelString()
                      << " | " << std::fixed << std::setprecision(2) << cycle.duration << "s"
                      << " | " << cycle.samples.size() << " samples"
                      << " | " << cycle.sourceFile << "\n";
        }
        if (dataset.size() > 5) {
            std::cout << "  ... and " << (dataset.size() - 5) << " more entries\n";
        }
    }
    
    return 0;
}

/**
 * @brief Test feature extraction trên một vài samples
 */
int testFeatureExtraction(const std::string& datasetPath) {
    std::cout << "\n[Mode] Feature Extraction Test\n";
    std::cout << "Path: " << datasetPath << "\n\n";
    
    // Load một vài samples
    DatasetManager manager(datasetPath);
    
    if (!manager.isValidDatasetPath()) {
        std::cerr << "Error: Invalid dataset path!\n";
        return 1;
    }
    
    auto wavFiles = manager.getWavFiles();
    if (wavFiles.empty()) {
        std::cerr << "Error: No WAV files found!\n";
        return 1;
    }
    
    // Xử lý 3 file đầu tiên
    std::vector<LabeledBreathingCycle> allCycles;
    size_t numFiles = std::min(wavFiles.size(), size_t(3));
    
    std::cout << "Loading " << numFiles << " files for feature extraction test...\n\n";
    
    for (size_t i = 0; i < numFiles; ++i) {
        std::vector<LabeledBreathingCycle> fileCycles;
        if (manager.processFile(wavFiles[i], fileCycles)) {
            std::cout << "  [" << (i + 1) << "] " << fs::path(wavFiles[i]).filename().string()
                      << " -> " << fileCycles.size() << " cycles\n";
            for (auto& cycle : fileCycles) {
                allCycles.push_back(std::move(cycle));
            }
        }
    }
    
    if (allCycles.empty()) {
        std::cerr << "Error: No cycles loaded!\n";
        return 1;
    }
    
    std::cout << "\nTotal cycles: " << allCycles.size() << "\n\n";
    
    // Khởi tạo FeatureExtractor
    FeatureExtractor extractor(FEATURE_SAMPLE_RATE);
    
    std::cout << "Feature Extractor Configuration:\n";
    std::cout << "  Sample rate:    " << extractor.getSampleRate() << " Hz\n";
    std::cout << "  Frame size:     " << extractor.getFrameSize() << " samples ("
              << (extractor.getFrameSize() * 1000.0f / extractor.getSampleRate()) << " ms)\n";
    std::cout << "  Hop size:       " << extractor.getHopSize() << " samples\n";
    std::cout << "  Feature dim:    " << CycleFeatures::getFeatureDimension() << "\n\n";
    
    // Extract features cho một vài cycles
    std::cout << "Extracting features for first 5 cycles...\n\n";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < std::min(allCycles.size(), size_t(5)); ++i) {
        const auto& cycle = allCycles[i];
        CycleFeatures features;
        
        if (extractor.extractFeatures(cycle.samples, features, false)) {
            std::cout << "═══════════════════════════════════════════════════════════════\n";
            std::cout << "Cycle " << (i + 1) << ": " << cycle.sourceFile 
                      << " [" << cycle.getLabelString() << "]\n";
            std::cout << "  Duration: " << std::fixed << std::setprecision(2) 
                      << features.durationSec << "s, Frames: " << features.numFrames << "\n";
            std::cout << "───────────────────────────────────────────────────────────────\n";
            std::cout << "  Time-Domain Features:\n";
            std::cout << "    EED:       " << std::scientific << std::setprecision(4) 
                      << features.eed << "\n";
            std::cout << "    ZCR:       " << std::fixed << std::setprecision(4) 
                      << features.zcr_mean << " ± " << features.zcr_std << "\n";
            std::cout << "    RMSE:      " << features.rmse_mean << " ± " << features.rmse_std << "\n";
            std::cout << "───────────────────────────────────────────────────────────────\n";
            std::cout << "  MFCC Features (first 5 coefficients):\n";
            std::cout << "    Static:    [";
            for (int c = 0; c < 5; ++c) {
                std::cout << std::setw(8) << std::setprecision(3) << features.mfcc_mean[c];
                if (c < 4) std::cout << ", ";
            }
            std::cout << ", ...]\n";
            std::cout << "    Delta:     [";
            for (int c = 0; c < 5; ++c) {
                std::cout << std::setw(8) << std::setprecision(3) << features.delta_mean[c];
                if (c < 4) std::cout << ", ";
            }
            std::cout << ", ...]\n";
            std::cout << "    Delta2:    [";
            for (int c = 0; c < 5; ++c) {
                std::cout << std::setw(8) << std::setprecision(3) << features.delta2_mean[c];
                if (c < 4) std::cout << ", ";
            }
            std::cout << ", ...]\n";
            
            // Flat vector
            auto flat = features.toFlatVector();
            std::cout << "───────────────────────────────────────────────────────────────\n";
            std::cout << "  Flat vector size: " << flat.size() << " features\n";
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "Feature extraction time: " << duration.count() << " ms\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    return 0;
}

/**
 * @brief Extract features và lưu ra CSV
 */
int extractAndSaveFeatures(const std::string& datasetPath, bool quickMode) {
    std::cout << "\n[Mode] Extract Features & Save to CSV\n";
    std::cout << "Path: " << datasetPath << "\n\n";
    
    DatasetManager manager(datasetPath);
    
    if (!manager.isValidDatasetPath()) {
        std::cerr << "Error: Invalid dataset path!\n";
        return 1;
    }
    
    // Load dataset
    std::vector<LabeledBreathingCycle> dataset;
    
    if (quickMode) {
        auto wavFiles = manager.getWavFiles();
        size_t numFiles = std::min(wavFiles.size(), size_t(20));
        std::cout << "Quick mode: Processing " << numFiles << " files...\n";
        
        for (size_t i = 0; i < numFiles; ++i) {
            std::vector<LabeledBreathingCycle> fileCycles;
            if (manager.processFile(wavFiles[i], fileCycles)) {
                for (auto& cycle : fileCycles) {
                    dataset.push_back(std::move(cycle));
                }
            }
            if ((i + 1) % 5 == 0) {
                std::cout << "  Processed " << (i + 1) << "/" << numFiles << " files\n";
            }
        }
    } else {
        std::cout << "Loading full dataset...\n";
        manager.loadDatasetWithProgress(dataset,
            [](size_t current, size_t total) {
                if (current % 100 == 0 || current == total) {
                    std::cout << "\r  Progress: " << current << "/" << total << "    " << std::flush;
                }
            });
        std::cout << "\n";
    }
    
    std::cout << "\nTotal cycles: " << dataset.size() << "\n";
    std::cout << "Extracting features...\n";
    
    // Extract features
    FeatureExtractor extractor(FEATURE_SAMPLE_RATE);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Mở file CSV để ghi
    std::ofstream csvFile("features.csv");
    if (!csvFile.is_open()) {
        std::cerr << "Error: Cannot create features.csv!\n";
        return 1;
    }
    
    // Ghi header
    csvFile << "patient_id,label,duration";
    auto featureNames = CycleFeatures::getFeatureNames();
    for (const auto& name : featureNames) {
        csvFile << "," << name;
    }
    csvFile << "\n";
    
    size_t successCount = 0;
    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& cycle = dataset[i];
        CycleFeatures features;
        
        if (extractor.extractFeatures(cycle.samples, features, false)) {
            // Ghi vào CSV
            csvFile << cycle.patientInfo.patientId << ","
                    << static_cast<int>(cycle.label) << ","
                    << std::fixed << std::setprecision(4) << cycle.duration;
            
            auto flat = features.toFlatVector();
            for (float val : flat) {
                csvFile << "," << std::setprecision(6) << val;
            }
            csvFile << "\n";
            
            successCount++;
        }
        
        if ((i + 1) % 500 == 0) {
            std::cout << "  Extracted " << (i + 1) << "/" << dataset.size() 
                      << " (" << successCount << " success)\n";
        }
    }
    
    csvFile.close();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "                   FEATURE EXTRACTION COMPLETE                  \n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Cycles processed: " << successCount << "/" << dataset.size() << "\n";
    std::cout << "  Feature dimension: " << CycleFeatures::getFeatureDimension() << "\n";
    std::cout << "  Output file: features.csv\n";
    std::cout << "  Time elapsed: " << duration.count() << " seconds\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    return 0;
}

/**
 * @brief Train và test Cascaded Classifier
 */
int testCascadedClassifier(const std::string& datasetPath, bool quickMode) {
    std::cout << "\n[Mode] Cascaded Classifier Training & Evaluation\n";
    std::cout << "Path: " << datasetPath << "\n\n";
    
    DatasetManager manager(datasetPath);
    
    if (!manager.isValidDatasetPath()) {
        std::cerr << "Error: Invalid dataset path!\n";
        return 1;
    }
    
    // Load dataset
    std::vector<LabeledBreathingCycle> dataset;
    
    if (quickMode) {
        auto wavFiles = manager.getWavFiles();
        size_t numFiles = std::min(wavFiles.size(), size_t(50));
        std::cout << "Quick mode: Processing " << numFiles << " files...\n";
        
        for (size_t i = 0; i < numFiles; ++i) {
            std::vector<LabeledBreathingCycle> fileCycles;
            if (manager.processFile(wavFiles[i], fileCycles)) {
                for (auto& cycle : fileCycles) {
                    dataset.push_back(std::move(cycle));
                }
            }
            if ((i + 1) % 10 == 0) {
                std::cout << "  Processed " << (i + 1) << "/" << numFiles << " files\n";
            }
        }
    } else {
        std::cout << "Loading full dataset...\n";
        manager.loadDatasetWithProgress(dataset,
            [](size_t current, size_t total) {
                if (current % 100 == 0 || current == total) {
                    std::cout << "\r  Progress: " << current << "/" << total << "    " << std::flush;
                }
            });
        std::cout << "\n";
    }
    
    std::cout << "\nTotal cycles: " << dataset.size() << "\n";
    
    if (dataset.size() < 20) {
        std::cerr << "Error: Not enough data for training!\n";
        return 1;
    }
    
    // Split train/test (80/20)
    std::vector<LabeledBreathingCycle> trainData, testData;
    DatasetManager::splitDataset(dataset, 0.8f, trainData, testData, 42);
    
    std::cout << "Train set: " << trainData.size() << " cycles\n";
    std::cout << "Test set: " << testData.size() << " cycles\n\n";
    
    // Initialize and train controller
    CascadedController controller;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    trainAndEvaluateCascaded(trainData, testData, controller, true);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "Total time: " << duration.count() << " seconds\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    
    // Demo prediction on a few samples
    std::cout << "\n[Demo Predictions]\n";
    
    FeatureExtractor extractor(FEATURE_SAMPLE_RATE);
    
    for (size_t i = 0; i < std::min(testData.size(), size_t(5)); ++i) {
        const auto& cycle = testData[i];
        CycleFeatures cf;
        
        if (extractor.extractFeatures(cycle.samples, cf, false)) {
            PatientMetadata meta;
            meta.patientId = cycle.patientInfo.patientId;
            
            FeatureVector fv = cycleFeaturesToFeatureVector(cf, meta);
            PredictionResult result = controller.predict(fv);
            
            std::cout << "  Sample " << (i + 1) << ": "
                      << "Actual=" << labelToString(cycle.label)
                      << ", Predicted=" << result.getClassName()
                      << ", Confidence=" << std::fixed << std::setprecision(3) << result.confidence
                      << ", ExitLayer=" << cascadeLayerToString(result.exitLayer)
                      << "\n";
        }
    }
    
    return 0;
}

/**
 * @brief Main function
 */
int main(int argc, char* argv[]) {
    std::string datasetPath = DEFAULT_DATASET_PATH;
    bool quickTest = false;
    bool showStats = false;
    bool testFeatures = false;
    bool extractFeatures = false;
    bool testClassifier = false;
    std::string singleFile;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--test") {
            quickTest = true;
        }
        else if (arg == "--features") {
            testFeatures = true;
        }
        else if (arg == "--extract") {
            extractFeatures = true;
        }
        else if (arg == "--classify") {
            testClassifier = true;
        }
        else if (arg == "--stats") {
            showStats = true;
        }
        else if (arg == "--path" && i + 1 < argc) {
            datasetPath = argv[++i];
        }
        else if (arg.find("--") != 0 && arg.find("-") != 0) {
            // Không phải option -> có thể là file path
            singleFile = arg;
        }
    }
    
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Respiratory Sound Analysis System v3.0                   ║\n";
    std::cout << "║     Cascaded Framework for FPGA (Ultra96-V2)                 ║\n";
    std::cout << "║     Phase 1: Preprocessing | Phase 2: Features | Phase 3: RF ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    // Nếu có file cụ thể -> xử lý file đó
    if (!singleFile.empty()) {
        return processSingleFile(singleFile);
    }
    
    // Test feature extraction
    if (testFeatures) {
        return testFeatureExtraction(datasetPath);
    }
    
    // Extract và lưu features
    if (extractFeatures) {
        return extractAndSaveFeatures(datasetPath, quickTest);
    }
    
    // Test Cascaded Classifier
    if (testClassifier) {
        return testCascadedClassifier(datasetPath, quickTest);
    }
    
    // Nếu chỉ cần stats -> load và hiển thị
    if (showStats) {
        std::cout << "\n[Mode] Statistics Only\n\n";
        DatasetManager manager(datasetPath);
        
        if (!manager.isValidDatasetPath()) {
            std::cerr << "Error: Invalid dataset path!\n";
            return 1;
        }
        
        std::vector<LabeledBreathingCycle> dataset;
        manager.loadDataset(dataset, true);
        return 0;
    }
    
    // Mặc định: xử lý dataset
    return processDataset(datasetPath, quickTest);
}
