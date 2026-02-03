/**
 * @file RandomForestCPU.cpp
 * @brief CPU-based Random Forest utilities and helpers
 * 
 * Các hàm phụ trợ cho Random Forest inference trên CPU.
 * Implementation chính đã nằm trong CascadedLogic.cpp
 * 
 * File này chứa:
 * - Training utilities
 * - Model evaluation
 * - Batch prediction helpers
 * 
 * @author Research Team
 * @date 2026
 */

#include "CascadedLogic.h"
#include "FeatureExtraction.h"
#include "SignalPrep.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cmath>

namespace respiratory {

// ============================================================================
// EVALUATION UTILITIES
// ============================================================================

/**
 * @brief Tính accuracy từ predictions và labels
 */
float computeAccuracy(const std::vector<PredictionClass>& predictions,
                      const std::vector<int>& labels) {
    if (predictions.size() != labels.size() || predictions.empty()) {
        return 0.0f;
    }
    
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (static_cast<int>(predictions[i]) == labels[i]) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / predictions.size();
}

/**
 * @brief In confusion matrix
 */
void printConfusionMatrix(const std::vector<PredictionClass>& predictions,
                          const std::vector<int>& labels) {
    // 4x4 matrix: Normal, Crackle, Wheeze, Both
    std::array<std::array<int, NUM_CLASSES>, NUM_CLASSES> matrix = {};
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred = static_cast<int>(predictions[i]);
        int actual = labels[i];
        
        if (pred >= 0 && pred < NUM_CLASSES && actual >= 0 && actual < NUM_CLASSES) {
            matrix[actual][pred]++;
        }
    }
    
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "              Pred_N  Pred_C  Pred_W  Pred_B\n";
    std::cout << "  Actual_N  ";
    for (int j = 0; j < NUM_CLASSES; ++j) {
        std::cout << std::setw(8) << matrix[0][j];
    }
    std::cout << "\n  Actual_C  ";
    for (int j = 0; j < NUM_CLASSES; ++j) {
        std::cout << std::setw(8) << matrix[1][j];
    }
    std::cout << "\n  Actual_W  ";
    for (int j = 0; j < NUM_CLASSES; ++j) {
        std::cout << std::setw(8) << matrix[2][j];
    }
    std::cout << "\n  Actual_B  ";
    for (int j = 0; j < NUM_CLASSES; ++j) {
        std::cout << std::setw(8) << matrix[3][j];
    }
    std::cout << "\n";
    
    // Per-class metrics
    std::cout << "\nPer-class metrics:\n";
    const char* classNames[] = {"Normal", "Crackle", "Wheeze", "Both"};
    
    for (int c = 0; c < NUM_CLASSES; ++c) {
        int tp = matrix[c][c];
        int fn = 0, fp = 0;
        
        for (int j = 0; j < NUM_CLASSES; ++j) {
            if (j != c) {
                fn += matrix[c][j];  // False negatives
                fp += matrix[j][c];  // False positives
            }
        }
        
        float precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
        float recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;
        float f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0f;
        
        std::cout << "  " << classNames[c] << ": "
                  << "Precision=" << std::fixed << std::setprecision(3) << precision
                  << ", Recall=" << recall
                  << ", F1=" << f1 << "\n";
    }
}

/**
 * @brief Chuyển đổi CycleFeatures sang FeatureVector cho classifier
 */
FeatureVector cycleFeaturesToFeatureVector(const CycleFeatures& cycleFeatures,
                                            const PatientMetadata& metadata) {
    FeatureVector fv;
    
    // Metadata
    fv.metadata = metadata;
    
    // Time-domain features
    fv.eed = cycleFeatures.eed;
    fv.zcr_mean = cycleFeatures.zcr_mean;
    fv.zcr_std = cycleFeatures.zcr_std;
    fv.rmse_mean = cycleFeatures.rmse_mean;
    fv.rmse_std = cycleFeatures.rmse_std;
    
    // Derived features
    fv.globalEnergy = cycleFeatures.rmse_mean * cycleFeatures.rmse_mean;
    fv.energyVariance = cycleFeatures.rmse_std * cycleFeatures.rmse_std;
    
    // MFCC features
    fv.mfcc_mean = cycleFeatures.mfcc_mean;
    fv.mfcc_std = cycleFeatures.mfcc_std;
    fv.delta_mean = cycleFeatures.delta_mean;
    fv.delta_std = cycleFeatures.delta_std;
    fv.delta2_mean = cycleFeatures.delta2_mean;
    fv.delta2_std = cycleFeatures.delta2_std;
    
    // Amplitude distribution (approximation)
    // Skewness và kurtosis có thể tính từ MFCC
    fv.amplitudeSkewness = 0.0f;  // Would need raw samples
    fv.amplitudeKurtosis = 3.0f;  // Normal distribution default
    
    return fv;
}

/**
 * @brief Train và evaluate Cascaded Controller
 */
void trainAndEvaluateCascaded(
    const std::vector<LabeledBreathingCycle>& trainData,
    const std::vector<LabeledBreathingCycle>& testData,
    CascadedController& controller,
    bool verbose) {
    
    if (verbose) {
        std::cout << "\n[Training Cascaded Controller]\n";
        std::cout << "  Train samples: " << trainData.size() << "\n";
        std::cout << "  Test samples: " << testData.size() << "\n";
    }
    
    // Extract features
    FeatureExtractor extractor(FEATURE_SAMPLE_RATE);
    
    std::vector<FeatureVector> trainFeatures;
    std::vector<int> trainLabels;
    
    for (const auto& cycle : trainData) {
        CycleFeatures cf;
        if (extractor.extractFeatures(cycle.samples, cf, false)) {
            PatientMetadata meta;
            meta.patientId = cycle.patientInfo.patientId;
            
            FeatureVector fv = cycleFeaturesToFeatureVector(cf, meta);
            trainFeatures.push_back(fv);
            trainLabels.push_back(static_cast<int>(cycle.label));
        }
    }
    
    if (verbose) {
        std::cout << "  Extracted " << trainFeatures.size() << " feature vectors\n";
    }
    
    // Train
    controller.trainAll(trainFeatures, trainLabels);
    
    if (verbose) {
        std::cout << "  Training complete.\n";
    }
    
    // Evaluate on test set
    if (!testData.empty()) {
        std::vector<FeatureVector> testFeatures;
        std::vector<int> testLabels;
        
        for (const auto& cycle : testData) {
            CycleFeatures cf;
            if (extractor.extractFeatures(cycle.samples, cf, false)) {
                PatientMetadata meta;
                meta.patientId = cycle.patientInfo.patientId;
                
                FeatureVector fv = cycleFeaturesToFeatureVector(cf, meta);
                testFeatures.push_back(fv);
                testLabels.push_back(static_cast<int>(cycle.label));
            }
        }
        
        // Predict
        std::vector<PredictionClass> predictions;
        for (const auto& fv : testFeatures) {
            auto result = controller.predict(fv);
            predictions.push_back(result.predictedClass);
        }
        
        // Metrics
        float accuracy = computeAccuracy(predictions, testLabels);
        
        if (verbose) {
            std::cout << "\n[Evaluation Results]\n";
            std::cout << "  Accuracy: " << std::fixed << std::setprecision(4) 
                      << accuracy << " (" << (accuracy * 100) << "%)\n";
            
            // Exit statistics
            auto stats = controller.getExitStatistics();
            std::cout << "\n[Early Exit Statistics]\n";
            std::cout << "  Layer 1 exits: " << stats.layer1Exits 
                      << " (" << (stats.getLayer1ExitRate() * 100) << "%)\n";
            std::cout << "  Layer 2 exits: " << stats.layer2Exits << "\n";
            std::cout << "  Layer 3 exits: " << stats.layer3Exits << "\n";
            std::cout << "  CNN fallbacks: " << stats.cnnFallbacks << "\n";
            
            printConfusionMatrix(predictions, testLabels);
        }
    }
}

} // namespace respiratory
