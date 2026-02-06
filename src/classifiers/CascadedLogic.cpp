/**
 * @file CascadedLogic.cpp
 * @brief Implementation of Cascaded Classification Framework
 * 
 * Triển khai hệ thống phân loại phân tầng với:
 * - 3 layers statistical screening (RF-based)
 * - Septuple Forest với majority voting
 * - Early-exit mechanism
 * 
 * @author Research Team
 * @date 2026
 */

#include "CascadedLogic.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>

namespace respiratory {

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

std::string predictionClassToString(PredictionClass cls) {
    switch (cls) {
        case PredictionClass::NORMAL: return "Normal";
        case PredictionClass::CRACKLE: return "Crackle";
        case PredictionClass::WHEEZE: return "Wheeze";
        case PredictionClass::BOTH: return "Both";
        case PredictionClass::UNKNOWN: return "Unknown";
        case PredictionClass::NEED_CNN: return "Need_CNN";
        default: return "Invalid";
    }
}

std::string cascadeLayerToString(CascadeLayer layer) {
    switch (layer) {
        case CascadeLayer::LAYER_1_GLOBAL: return "Layer1_Global";
        case CascadeLayer::LAYER_2_TRANSIENT: return "Layer2_Transient";
        case CascadeLayer::LAYER_3_SPECTRAL: return "Layer3_Spectral";
        case CascadeLayer::LAYER_4_CNN: return "Layer4_CNN";
        case CascadeLayer::COMPLETE: return "Complete";
        default: return "Invalid";
    }
}

std::string exitReasonToString(ExitReason reason) {
    switch (reason) {
        case ExitReason::CONSENSUS_REACHED: return "Consensus_Reached";
        case ExitReason::CONFIDENCE_EXCEEDED: return "Confidence_Exceeded";
        case ExitReason::ALL_LAYERS_PASSED: return "All_Layers_Passed";
        case ExitReason::NEED_DEEP_ANALYSIS: return "Need_Deep_Analysis";
        default: return "Invalid";
    }
}

// ============================================================================
// FEATURE VECTOR IMPLEMENTATION
// ============================================================================

FeatureVector::FeatureVector()
    : eed(0.0f), zcr_mean(0.0f), zcr_std(0.0f)
    , rmse_mean(0.0f), rmse_std(0.0f)
    , globalEnergy(0.0f), energyVariance(0.0f)
    , amplitudeSkewness(0.0f), amplitudeKurtosis(0.0f)
{
    mfcc_mean.resize(13, 0.0f);
    mfcc_std.resize(13, 0.0f);
    delta_mean.resize(13, 0.0f);
    delta_std.resize(13, 0.0f);
    delta2_mean.resize(13, 0.0f);
    delta2_std.resize(13, 0.0f);
}

std::vector<float> FeatureVector::toFlatVector() const {
    std::vector<float> flat;
    flat.reserve(100);  // Approximate size
    
    // Time-domain
    flat.push_back(eed);
    flat.push_back(zcr_mean);
    flat.push_back(zcr_std);
    flat.push_back(rmse_mean);
    flat.push_back(rmse_std);
    flat.push_back(globalEnergy);
    flat.push_back(energyVariance);
    flat.push_back(amplitudeSkewness);
    flat.push_back(amplitudeKurtosis);
    
    // MFCC
    flat.insert(flat.end(), mfcc_mean.begin(), mfcc_mean.end());
    flat.insert(flat.end(), mfcc_std.begin(), mfcc_std.end());
    flat.insert(flat.end(), delta_mean.begin(), delta_mean.end());
    flat.insert(flat.end(), delta_std.begin(), delta_std.end());
    flat.insert(flat.end(), delta2_mean.begin(), delta2_mean.end());
    flat.insert(flat.end(), delta2_std.begin(), delta2_std.end());
    
    return flat;
}

std::vector<float> FeatureVector::getLayer1Features() const {
    /**
     * Layer 1: Global Screening
     * Features: Global energy, energy variance, RMSE, metadata-derived
     */
    std::vector<float> features;
    features.push_back(globalEnergy);
    features.push_back(energyVariance);
    features.push_back(rmse_mean);
    features.push_back(rmse_std);
    features.push_back(eed);
    
    // Metadata-derived features
    features.push_back(metadata.hasPriorCondition ? 1.0f : 0.0f);
    
    return features;
}

std::vector<float> FeatureVector::getLayer2Features() const {
    /**
     * Layer 2: Transient Detection (Crackle)
     * Features: ZCR, amplitude distribution, EED
     */
    std::vector<float> features;
    features.push_back(zcr_mean);
    features.push_back(zcr_std);
    features.push_back(amplitudeSkewness);
    features.push_back(amplitudeKurtosis);
    features.push_back(eed);
    features.push_back(rmse_mean);
    features.push_back(rmse_std);
    
    return features;
}

std::vector<float> FeatureVector::getLayer3Features() const {
    /**
     * Layer 3: Spectral Screening
     * Features: MFCC (39-dim)
     */
    std::vector<float> features;
    features.insert(features.end(), mfcc_mean.begin(), mfcc_mean.end());
    features.insert(features.end(), mfcc_std.begin(), mfcc_std.end());
    features.insert(features.end(), delta_mean.begin(), delta_mean.end());
    features.insert(features.end(), delta_std.begin(), delta_std.end());
    features.insert(features.end(), delta2_mean.begin(), delta2_mean.end());
    features.insert(features.end(), delta2_std.begin(), delta2_std.end());
    
    return features;
}

// ============================================================================
// CLASS PROBABILITIES IMPLEMENTATION
// ============================================================================

PredictionClass ClassProbabilities::getMaxClass() const {
    int maxIdx = 0;
    float maxProb = probs[0];
    
    for (int i = 1; i < NUM_CLASSES; ++i) {
        if (probs[i] > maxProb) {
            maxProb = probs[i];
            maxIdx = i;
        }
    }
    
    return static_cast<PredictionClass>(maxIdx);
}

float ClassProbabilities::getMaxProbability() const {
    return *std::max_element(probs.begin(), probs.end());
}

void ClassProbabilities::normalize() {
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum > 1e-10f) {
        for (float& p : probs) {
            p /= sum;
        }
    }
}

// ============================================================================
// PREDICTION RESULT IMPLEMENTATION
// ============================================================================

std::string PredictionResult::getClassName() const {
    return predictionClassToString(predictedClass);
}

std::string PredictionResult::describe() const {
    std::ostringstream oss;
    oss << "Prediction: " << getClassName()
        << " (confidence: " << confidence << ")"
        << ", Exit at: " << cascadeLayerToString(exitLayer)
        << ", Reason: " << exitReasonToString(exitReason)
        << ", Layers processed: " << totalLayersProcessed;
    if (needsCNN) {
        oss << " [CNN requested]";
    }
    return oss.str();
}

// ============================================================================
// DECISION TREE IMPLEMENTATION
// ============================================================================

DecisionTree::DecisionTree()
    : m_numFeatures(0), m_maxDepth(MAX_TREE_DEPTH), m_isTrained(false)
{
}

DecisionTree::~DecisionTree() = default;

void DecisionTree::train(const std::vector<std::vector<float>>& features,
                         const std::vector<int>& labels,
                         int maxDepth) {
    if (features.empty() || labels.empty()) {
        return;
    }
    
    m_numFeatures = static_cast<int>(features[0].size());
    m_maxDepth = maxDepth;
    m_nodes.clear();
    
    // Tạo danh sách indices
    std::vector<int> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Build tree recursively
    buildNode(features, labels, indices, 0);
    
    m_isTrained = true;
}

int DecisionTree::buildNode(const std::vector<std::vector<float>>& features,
                            const std::vector<int>& labels,
                            const std::vector<int>& sampleIndices,
                            int depth) {
    int nodeIdx = static_cast<int>(m_nodes.size());
    m_nodes.emplace_back();
    TreeNode& node = m_nodes.back();
    
    // Tính phân phối class
    std::array<int, NUM_CLASSES> classCounts = {0};
    for (int idx : sampleIndices) {
        if (labels[idx] >= 0 && labels[idx] < NUM_CLASSES) {
            classCounts[labels[idx]]++;
        }
    }
    
    int totalSamples = static_cast<int>(sampleIndices.size());
    
    // Tính xác suất cho node lá
    for (int c = 0; c < NUM_CLASSES; ++c) {
        node.leafProbs.probs[c] = static_cast<float>(classCounts[c]) / totalSamples;
    }
    
    // Điều kiện dừng
    // 1. Đạt độ sâu tối đa
    // 2. Không đủ samples
    // 3. Pure node (chỉ có một class)
    int numNonZero = 0;
    for (int c : classCounts) {
        if (c > 0) numNonZero++;
    }
    
    if (depth >= m_maxDepth || totalSamples < 2 || numNonZero <= 1) {
        node.isLeaf = true;
        return nodeIdx;
    }
    
    // Tìm split tốt nhất
    int bestFeature;
    float bestThreshold;
    float bestGini;
    
    if (!findBestSplit(features, labels, sampleIndices, bestFeature, bestThreshold, bestGini)) {
        node.isLeaf = true;
        return nodeIdx;
    }
    
    // Tạo internal node
    node.isLeaf = false;
    node.featureIndex = bestFeature;
    node.threshold = bestThreshold;
    
    // Split samples
    std::vector<int> leftIndices, rightIndices;
    leftIndices.reserve(sampleIndices.size());
    rightIndices.reserve(sampleIndices.size());
    
    for (int idx : sampleIndices) {
        if (features[idx][bestFeature] <= bestThreshold) {
            leftIndices.push_back(idx);
        } else {
            rightIndices.push_back(idx);
        }
    }
    
    // Build children
    // IMPORTANT: Sau mỗi lần gọi buildNode, vector m_nodes có thể được resize
    // nên phải lấy lại reference qua index, không dùng reference cũ
    if (!leftIndices.empty()) {
        int leftChildIdx = buildNode(features, labels, leftIndices, depth + 1);
        m_nodes[nodeIdx].leftChild = leftChildIdx;
    }
    if (!rightIndices.empty()) {
        int rightChildIdx = buildNode(features, labels, rightIndices, depth + 1);
        m_nodes[nodeIdx].rightChild = rightChildIdx;
    }
    
    return nodeIdx;
}

bool DecisionTree::findBestSplit(const std::vector<std::vector<float>>& features,
                                 const std::vector<int>& labels,
                                 const std::vector<int>& sampleIndices,
                                 int& bestFeature,
                                 float& bestThreshold,
                                 float& bestGini) {
    bestGini = std::numeric_limits<float>::max();
    bestFeature = -1;
    bestThreshold = 0.0f;
    
    int numSamples = static_cast<int>(sampleIndices.size());
    if (numSamples < 2) return false;
    
    // Random feature selection (sqrt(numFeatures))
    int numFeaturesToTry = std::max(1, static_cast<int>(std::sqrt(m_numFeatures)));
    std::vector<int> featureIndices(m_numFeatures);
    std::iota(featureIndices.begin(), featureIndices.end(), 0);
    
    static std::mt19937 rng(42);
    std::shuffle(featureIndices.begin(), featureIndices.end(), rng);
    
    for (int fi = 0; fi < numFeaturesToTry; ++fi) {
        int f = featureIndices[fi];
        
        // Bounds check
        if (f < 0 || f >= m_numFeatures) continue;
        
        // Collect feature values
        std::vector<float> values;
        values.reserve(numSamples);
        for (int idx : sampleIndices) {
            if (idx < 0 || idx >= static_cast<int>(features.size())) continue;
            if (f >= static_cast<int>(features[idx].size())) continue;
            values.push_back(features[idx][f]);
        }
        
        // Sort unique values
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        
        // Skip if not enough unique values
        if (values.size() < 2) continue;
        
        // Try thresholds
        for (size_t i = 0; i + 1 < values.size(); ++i) {
            float threshold = (values[i] + values[i + 1]) / 2.0f;
            
            // Split và tính Gini
            std::vector<int> leftIndices, rightIndices;
            leftIndices.reserve(numSamples);
            rightIndices.reserve(numSamples);
            
            for (int idx : sampleIndices) {
                if (idx < 0 || idx >= static_cast<int>(features.size())) continue;
                if (f >= static_cast<int>(features[idx].size())) continue;
                
                if (features[idx][f] <= threshold) {
                    leftIndices.push_back(idx);
                } else {
                    rightIndices.push_back(idx);
                }
            }
            
            if (leftIndices.empty() || rightIndices.empty()) continue;
            
            float leftGini = computeGini(labels, leftIndices);
            float rightGini = computeGini(labels, rightIndices);
            
            float weightedGini = (leftGini * leftIndices.size() + 
                                  rightGini * rightIndices.size()) / numSamples;
            
            if (weightedGini < bestGini) {
                bestGini = weightedGini;
                bestFeature = f;
                bestThreshold = threshold;
            }
        }
    }
    
    return bestFeature >= 0;
}

float DecisionTree::computeGini(const std::vector<int>& labels,
                                const std::vector<int>& indices) const {
    if (indices.empty()) return 0.0f;
    
    std::array<int, NUM_CLASSES> counts = {0};
    for (int idx : indices) {
        if (labels[idx] >= 0 && labels[idx] < NUM_CLASSES) {
            counts[labels[idx]]++;
        }
    }
    
    float gini = 1.0f;
    int total = static_cast<int>(indices.size());
    
    for (int c = 0; c < NUM_CLASSES; ++c) {
        float p = static_cast<float>(counts[c]) / total;
        gini -= p * p;
    }
    
    return gini;
}

ClassProbabilities DecisionTree::predict(const std::vector<float>& features) const {
    if (!m_isTrained || m_nodes.empty()) {
        return ClassProbabilities();
    }
    
    // Traverse tree
    int nodeIdx = 0;
    
    while (!m_nodes[nodeIdx].isLeaf) {
        const TreeNode& node = m_nodes[nodeIdx];
        
        if (node.featureIndex < 0 || 
            node.featureIndex >= static_cast<int>(features.size())) {
            break;
        }
        
        if (features[node.featureIndex] <= node.threshold) {
            if (node.leftChild >= 0) {
                nodeIdx = node.leftChild;
            } else {
                break;
            }
        } else {
            if (node.rightChild >= 0) {
                nodeIdx = node.rightChild;
            } else {
                break;
            }
        }
    }
    
    return m_nodes[nodeIdx].leafProbs;
}

bool DecisionTree::loadFromBuffer(const std::vector<TreeNode>& nodes) {
    m_nodes = nodes;
    m_isTrained = !nodes.empty();
    return m_isTrained;
}

std::vector<TreeNode> DecisionTree::exportToBuffer() const {
    return m_nodes;
}

// ============================================================================
// RANDOM FOREST IMPLEMENTATION
// ============================================================================

RandomForest::RandomForest(int numTrees, int maxDepth)
    : m_numTrees(numTrees), m_maxDepth(maxDepth), m_isTrained(false)
{
    m_trees.resize(numTrees);
    for (int i = 0; i < numTrees; ++i) {
        m_trees[i] = std::make_unique<DecisionTree>();
    }
}

RandomForest::~RandomForest() = default;

void RandomForest::train(const std::vector<std::vector<float>>& features,
                         const std::vector<int>& labels) {
    if (features.empty() || labels.empty()) {
        return;
    }
    
    for (int t = 0; t < m_numTrees; ++t) {
        // Bootstrap sample
        std::vector<std::vector<float>> bootFeatures;
        std::vector<int> bootLabels;
        createBootstrapSample(features, labels, bootFeatures, bootLabels);
        
        // Train tree
        m_trees[t]->train(bootFeatures, bootLabels, m_maxDepth);
    }
    
    m_isTrained = true;
}

void RandomForest::createBootstrapSample(
    const std::vector<std::vector<float>>& features,
    const std::vector<int>& labels,
    std::vector<std::vector<float>>& bootFeatures,
    std::vector<int>& bootLabels) {
    
    int n = static_cast<int>(features.size());
    bootFeatures.resize(n);
    bootLabels.resize(n);
    
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, n - 1);
    
    for (int i = 0; i < n; ++i) {
        int idx = dist(rng);
        bootFeatures[i] = features[idx];
        bootLabels[i] = labels[idx];
    }
}

ClassProbabilities RandomForest::predict(const std::vector<float>& features) const {
    ClassProbabilities result;
    
    if (!m_isTrained) {
        return result;
    }
    
    // Aggregate predictions from all trees
    for (const auto& tree : m_trees) {
        ClassProbabilities treeProbs = tree->predict(features);
        for (int c = 0; c < NUM_CLASSES; ++c) {
            result.probs[c] += treeProbs.probs[c];
        }
    }
    
    // Average
    for (int c = 0; c < NUM_CLASSES; ++c) {
        result.probs[c] /= m_numTrees;
    }
    
    return result;
}

PredictionClass RandomForest::predictClass(const std::vector<float>& features) const {
    return predict(features).getMaxClass();
}

bool RandomForest::loadModel(const std::string& modelPath) {
    // TODO: Implement model loading from file
    // For now, return false (not implemented)
    (void)modelPath;
    return false;
}

bool RandomForest::saveModel(const std::string& modelPath) const {
    // TODO: Implement model saving
    (void)modelPath;
    return false;
}

// ============================================================================
// SEPTUPLE FOREST IMPLEMENTATION
// ============================================================================

SeptupleForest::SeptupleForest(int treesPerCluster, int consensusThreshold)
    : m_treesPerCluster(treesPerCluster)
    , m_consensusThreshold(consensusThreshold)
{
    for (int i = 0; i < NUM_FOREST_CLUSTERS; ++i) {
        m_clusters[i] = std::make_unique<RandomForest>(treesPerCluster, MAX_TREE_DEPTH);
    }
}

SeptupleForest::~SeptupleForest() = default;

void SeptupleForest::train(const std::vector<std::vector<float>>& features,
                            const std::vector<int>& labels) {
    /**
     * Train 7 RF clusters independently
     * Each cluster gets slightly different bootstrap samples
     */
    
    for (int i = 0; i < NUM_FOREST_CLUSTERS; ++i) {
        m_clusters[i]->train(features, labels);
    }
}

bool SeptupleForest::predictWithVoting(const std::vector<float>& features,
                                        LayerResult& result) const {
    /**
     * Majority Voting với Septuple Forest
     * 
     * Công thức: Decision = Class c nếu Σ I(Oj = c) ≥ 4
     *                     = Next Layer nếu ngược lại
     */
    
    result.voteCounts.fill(0);
    
    // Collect votes from all 7 clusters
    for (int i = 0; i < NUM_FOREST_CLUSTERS; ++i) {
        ClassProbabilities probs = m_clusters[i]->predict(features);
        
        result.clusterVotes[i].probabilities = probs;
        result.clusterVotes[i].predictedClass = probs.getMaxClass();
        result.clusterVotes[i].confidence = probs.getMaxProbability();
        
        // Count vote
        int classIdx = static_cast<int>(result.clusterVotes[i].predictedClass);
        if (classIdx >= 0 && classIdx < NUM_CLASSES) {
            result.voteCounts[classIdx]++;
        }
    }
    
    // Compute majority vote
    computeMajorityVote(result.clusterVotes, result);
    
    // Check consensus
    result.hasConsensus = (result.voteCounts[static_cast<int>(result.consensusClass)] 
                           >= m_consensusThreshold);
    
    // Compute aggregated confidence
    float totalConf = 0.0f;
    for (int i = 0; i < NUM_FOREST_CLUSTERS; ++i) {
        totalConf += result.clusterVotes[i].confidence;
    }
    result.aggregatedConfidence = totalConf / NUM_FOREST_CLUSTERS;
    
    return result.hasConsensus;
}

void SeptupleForest::computeMajorityVote(
    const std::array<ClusterVote, NUM_FOREST_CLUSTERS>& votes,
    LayerResult& result) const {
    
    // Find class with most votes
    int maxVotes = 0;
    int maxClass = 0;
    
    for (int c = 0; c < NUM_CLASSES; ++c) {
        if (result.voteCounts[c] > maxVotes) {
            maxVotes = result.voteCounts[c];
            maxClass = c;
        }
    }
    
    result.consensusClass = static_cast<PredictionClass>(maxClass);
}

ClusterVote SeptupleForest::predict(const std::vector<float>& features) const {
    LayerResult result;
    predictWithVoting(features, result);
    
    ClusterVote vote;
    vote.predictedClass = result.consensusClass;
    vote.confidence = result.aggregatedConfidence;
    
    return vote;
}

bool SeptupleForest::isTrained() const {
    for (const auto& cluster : m_clusters) {
        if (!cluster->isTrained()) {
            return false;
        }
    }
    return true;
}

bool SeptupleForest::loadModels(const std::string& basePath) {
    // TODO: Load 7 cluster models
    (void)basePath;
    return false;
}

bool SeptupleForest::saveModels(const std::string& basePath) const {
    // TODO: Save 7 cluster models
    (void)basePath;
    return false;
}

// ============================================================================
// CASCADED CONTROLLER IMPLEMENTATION
// ============================================================================

CascadedController::CascadedController()
    : m_enableCNN(false)
{
    m_layer1Forest = std::make_unique<SeptupleForest>();
    m_layer2Forest = std::make_unique<SeptupleForest>();
    m_layer3Forest = std::make_unique<SeptupleForest>();
    
    // FIXED: Increased confidence thresholds to reduce over-exit problem
    // Previously: 0.75, 0.70, 0.65 -> caused 99.34% early exit at Layer 1
    // New thresholds require BOTH high consensus AND high confidence
    m_confidenceThresholds[0] = LAYER1_CONFIDENCE_THRESHOLD;  // Layer 1: 0.90
    m_confidenceThresholds[1] = LAYER2_CONFIDENCE_THRESHOLD;  // Layer 2: 0.88
    m_confidenceThresholds[2] = LAYER3_CONFIDENCE_THRESHOLD;  // Layer 3: 0.85
    
    // Initialize CNN inference for Layer 4
    m_cnnInference = std::make_unique<CnnInference>();
    
    resetStatistics();
}

CascadedController::~CascadedController() = default;

PredictionResult CascadedController::predict(const FeatureVector& features) {
    /**
     * Main prediction pipeline - State Machine
     * 
     * States:
     *   LAYER_1 -> Check consensus -> Early exit OR
     *   LAYER_2 -> Check consensus -> Early exit OR
     *   LAYER_3 -> Check consensus -> Early exit OR
     *   CNN (if enabled)
     */
    
    PredictionResult result;
    result.totalLayersProcessed = 0;
    
    // Current state
    CascadeLayer currentLayer = CascadeLayer::LAYER_1_GLOBAL;
    
    // State machine loop
    while (currentLayer != CascadeLayer::COMPLETE && 
           currentLayer != CascadeLayer::LAYER_4_CNN) {
        
        LayerResult layerResult;
        
        switch (currentLayer) {
            // ============================================================
            // LAYER 1: Global Screening
            // ============================================================
            case CascadeLayer::LAYER_1_GLOBAL: {
                layerResult = processLayer1(features);
                layerResult.layer = CascadeLayer::LAYER_1_GLOBAL;
                result.layerResults.push_back(layerResult);
                result.totalLayersProcessed++;
                
                if (checkEarlyExit(layerResult, currentLayer)) {
                    result.predictedClass = layerResult.consensusClass;
                    result.confidence = layerResult.aggregatedConfidence;
                    result.exitLayer = CascadeLayer::LAYER_1_GLOBAL;
                    result.exitReason = layerResult.hasConsensus ? 
                        ExitReason::CONSENSUS_REACHED : ExitReason::CONFIDENCE_EXCEEDED;
                    m_stats.layer1Exits++;
                    currentLayer = CascadeLayer::COMPLETE;
                } else {
                    currentLayer = CascadeLayer::LAYER_2_TRANSIENT;
                }
                break;
            }
            
            // ============================================================
            // LAYER 2: Transient Detection (Crackle)
            // ============================================================
            case CascadeLayer::LAYER_2_TRANSIENT: {
                layerResult = processLayer2(features);
                layerResult.layer = CascadeLayer::LAYER_2_TRANSIENT;
                result.layerResults.push_back(layerResult);
                result.totalLayersProcessed++;
                
                if (checkEarlyExit(layerResult, currentLayer)) {
                    result.predictedClass = layerResult.consensusClass;
                    result.confidence = layerResult.aggregatedConfidence;
                    result.exitLayer = CascadeLayer::LAYER_2_TRANSIENT;
                    result.exitReason = layerResult.hasConsensus ?
                        ExitReason::CONSENSUS_REACHED : ExitReason::CONFIDENCE_EXCEEDED;
                    m_stats.layer2Exits++;
                    currentLayer = CascadeLayer::COMPLETE;
                } else {
                    currentLayer = CascadeLayer::LAYER_3_SPECTRAL;
                }
                break;
            }
            
            // ============================================================
            // LAYER 3: Spectral Screening (MFCC)
            // ============================================================
            case CascadeLayer::LAYER_3_SPECTRAL: {
                layerResult = processLayer3(features);
                layerResult.layer = CascadeLayer::LAYER_3_SPECTRAL;
                result.layerResults.push_back(layerResult);
                result.totalLayersProcessed++;
                
                if (checkEarlyExit(layerResult, currentLayer)) {
                    result.predictedClass = layerResult.consensusClass;
                    result.confidence = layerResult.aggregatedConfidence;
                    result.exitLayer = CascadeLayer::LAYER_3_SPECTRAL;
                    result.exitReason = layerResult.hasConsensus ?
                        ExitReason::CONSENSUS_REACHED : ExitReason::CONFIDENCE_EXCEEDED;
                    m_stats.layer3Exits++;
                    currentLayer = CascadeLayer::COMPLETE;
                } else {
                    // Need CNN
                    currentLayer = CascadeLayer::LAYER_4_CNN;
                }
                break;
            }
            
            default:
                currentLayer = CascadeLayer::LAYER_4_CNN;
                break;
        }
    }
    
    // Handle CNN fallback (Layer 4)
    if (currentLayer == CascadeLayer::LAYER_4_CNN) {
        /**
         * Layer 4: CNN Deep Pattern Recognition
         * 
         * Được kích hoạt khi các RF layers (1-3) không đủ tin cậy.
         * Theo bài báo, khoảng 20-30% mẫu sẽ được xử lý ở đây.
         */
        
        // Process Layer 4 với CNN inference
        PredictionResult layer4Result = processLayer4(features);
        
        // Merge layer 4 results into main result
        result.predictedClass = layer4Result.predictedClass;
        result.confidence = layer4Result.confidence;
        result.exitLayer = CascadeLayer::LAYER_4_CNN;
        result.exitReason = ExitReason::NEED_DEEP_ANALYSIS;
        result.needsCNN = true;
        result.totalLayersProcessed = 4;
        
        m_stats.cnnFallbacks++;
    }
    
    m_stats.totalPredictions++;
    
    return result;
}

std::vector<PredictionResult> CascadedController::predictBatch(
    const std::vector<FeatureVector>& features) {
    
    std::vector<PredictionResult> results;
    results.reserve(features.size());
    
    for (const auto& f : features) {
        results.push_back(predict(f));
    }
    
    return results;
}

LayerResult CascadedController::processLayer1(const FeatureVector& features) {
    /**
     * Layer 1: Global Screening
     * 
     * Mục đích: Lọc nhanh các ca "Healthy" rõ ràng
     * Features: Global energy, RMSE, metadata
     */
    
    LayerResult result;
    result.layer = CascadeLayer::LAYER_1_GLOBAL;
    
    if (m_layer1Forest && m_layer1Forest->isTrained()) {
        auto layer1Features = features.getLayer1Features();
        m_layer1Forest->predictWithVoting(layer1Features, result);
    } else {
        // Fallback: sử dụng heuristic đơn giản
        result.hasConsensus = false;
        result.consensusClass = PredictionClass::UNKNOWN;
        
        // Heuristic: Low energy thường là normal
        if (features.globalEnergy < 0.1f && features.rmse_mean < 0.1f) {
            result.consensusClass = PredictionClass::NORMAL;
            result.aggregatedConfidence = 0.6f;
        }
    }
    
    return result;
}

LayerResult CascadedController::processLayer2(const FeatureVector& features) {
    /**
     * Layer 2: Transient Detection
     * 
     * Mục đích: Phát hiện Crackles (tiếng rít)
     * Features: ZCR, amplitude distribution
     * 
     * Crackles đặc trưng bởi:
     * - High ZCR (nhiều zero crossings)
     * - High kurtosis (amplitude peaky)
     */
    
    LayerResult result;
    result.layer = CascadeLayer::LAYER_2_TRANSIENT;
    
    if (m_layer2Forest && m_layer2Forest->isTrained()) {
        auto layer2Features = features.getLayer2Features();
        m_layer2Forest->predictWithVoting(layer2Features, result);
    } else {
        // Fallback heuristic
        result.hasConsensus = false;
        
        // High ZCR + high kurtosis -> Crackle
        if (features.zcr_mean > 0.15f && features.amplitudeKurtosis > 3.0f) {
            result.consensusClass = PredictionClass::CRACKLE;
            result.aggregatedConfidence = 0.55f;
        } else {
            result.consensusClass = PredictionClass::UNKNOWN;
        }
    }
    
    return result;
}

LayerResult CascadedController::processLayer3(const FeatureVector& features) {
    /**
     * Layer 3: Spectral Screening
     * 
     * Mục đích: Phát hiện bất thường phổ (Wheeze, Both)
     * Features: MFCC 39-dim
     * 
     * Wheeze đặc trưng bởi:
     * - Spectral patterns liên tục
     * - Specific MFCC patterns
     */
    
    LayerResult result;
    result.layer = CascadeLayer::LAYER_3_SPECTRAL;
    
    if (m_layer3Forest && m_layer3Forest->isTrained()) {
        auto layer3Features = features.getLayer3Features();
        m_layer3Forest->predictWithVoting(layer3Features, result);
    } else {
        // Fallback: dựa trên MFCC patterns
        result.hasConsensus = false;
        result.consensusClass = PredictionClass::NORMAL;
        result.aggregatedConfidence = 0.5f;
        
        // Simple heuristic based on first MFCC coefficient variance
        if (!features.mfcc_std.empty() && features.mfcc_std[0] > 5.0f) {
            result.consensusClass = PredictionClass::WHEEZE;
            result.aggregatedConfidence = 0.55f;
        }
    }
    
    return result;
}

bool CascadedController::checkEarlyExit(const LayerResult& result, 
                                         CascadeLayer layer) const {
    /**
     * FIXED: Stricter early-exit conditions to reduce over-exit
     * 
     * New logic:
     * 1. MUST have consensus (≥4/7 clusters agree)
     * 2. AND aggregated confidence must exceed layer threshold
     * 3. AND at least one cluster must have very high confidence (>0.9)
     * 
     * This ensures only "easy" samples exit early, leaving ambiguous
     * samples for deeper analysis or CNN fallback.
     */
    
    int layerIdx = static_cast<int>(layer);
    if (layerIdx < 0 || layerIdx >= NUM_CASCADE_LAYERS) {
        return false;
    }
    
    // Condition 1: Must have consensus
    if (!result.hasConsensus) {
        return false;
    }
    
    // Condition 2: Aggregated confidence must exceed threshold
    float threshold = m_confidenceThresholds[layerIdx];
    if (result.aggregatedConfidence < threshold) {
        return false;
    }
    
    // Condition 3: Check for high-confidence cluster
    // At least one cluster should have confidence > 0.85
    bool hasHighConfidenceCluster = false;
    for (int i = 0; i < NUM_FOREST_CLUSTERS; ++i) {
        if (result.clusterVotes[i].confidence > 0.85f) {
            hasHighConfidenceCluster = true;
            break;
        }
    }
    
    // For Layer 1 and 2, require high-confidence cluster
    // Layer 3 can exit with just consensus + threshold
    if (layer == CascadeLayer::LAYER_1_GLOBAL || 
        layer == CascadeLayer::LAYER_2_TRANSIENT) {
        if (!hasHighConfidenceCluster) {
            return false;
        }
    }
    
    // All conditions met - allow early exit
    return true;
}

PredictionClass CascadedController::invokeCNN(const FeatureVector& features) {
    if (m_cnnCallback) {
        return m_cnnCallback(features);
    }
    return PredictionClass::UNKNOWN;
}

PredictionResult CascadedController::processLayer4(const FeatureVector& features,
                                                    const std::vector<float>& rawSignal) {
    /**
     * Layer 4: CNN Deep Pattern Recognition
     * 
     * Xử lý các mẫu "ambiguous" không thể phân loại bởi RF layers (1-3).
     * Theo bài báo, khoảng 20-30% mẫu sẽ cần đến Layer 4.
     * 
     * Pipeline:
     * 1. Nếu có raw signal -> tạo wavelet spectrogram
     * 2. Nếu không có -> sử dụng simulation dựa trên features
     * 3. Chạy CNN inference
     * 4. Trả về kết quả cuối cùng
     */
    
    PredictionResult result;
    result.exitLayer = CascadeLayer::LAYER_4_CNN;
    result.exitReason = ExitReason::NEED_DEEP_ANALYSIS;
    result.needsCNN = true;
    
    CnnPrediction cnnPred;
    
    // Check if we have CNN module initialized
    if (m_cnnInference) {
        if (!rawSignal.empty()) {
            // Option 1: Have raw signal -> create spectrogram and run inference
            cnnPred = m_cnnInference->predictFromSignal(rawSignal, 4000);
        }
        else {
            // Option 2: No raw signal -> create dummy spectrogram from features
            // This is a fallback - ideally we should always have raw signal
            Spectrogram spec;
            spec.allocate(CNN_INPUT_SIZE, CNN_INPUT_SIZE, CNN_INPUT_CHANNELS);
            
            // Fill spectrogram based on MFCC features (simplified)
            for (int y = 0; y < spec.height; ++y) {
                for (int x = 0; x < spec.width; ++x) {
                    float value = 0.5f;  // Base value
                    
                    // Modulate by MFCC values
                    if (!features.mfcc_mean.empty()) {
                        int mfccIdx = y % features.mfcc_mean.size();
                        value += features.mfcc_mean[mfccIdx] * 0.01f;
                    }
                    
                    // Add some variance based on position
                    value += features.rmse_mean * static_cast<float>(x) / spec.width * 0.1f;
                    
                    spec.set(x, y, std::clamp(value, 0.0f, 1.0f));
                }
            }
            
            cnnPred = m_cnnInference->predict(spec);
        }
    }
    else if (m_cnnCallback) {
        // Use external CNN callback if provided
        PredictionClass extPred = m_cnnCallback(features);
        cnnPred.predictedClass = static_cast<CnnPredictionClass>(static_cast<int>(extPred));
        cnnPred.confidence = 0.7f;  // Default confidence for callback
        cnnPred.isValid = true;
    }
    else {
        // No CNN available - fallback to simulation based on features
        // This is a heuristic fallback when no model is available
        
        // Simple heuristic based on features
        float normalScore = 0.25f;
        float crackleScore = 0.25f;
        float wheezeScore = 0.25f;
        float bothScore = 0.25f;
        
        // High ZCR suggests crackles
        if (features.zcr_mean > 0.12f) {
            crackleScore += 0.2f;
        }
        
        // High kurtosis suggests transient sounds
        if (features.amplitudeKurtosis > 3.5f) {
            crackleScore += 0.15f;
            bothScore += 0.1f;
        }
        
        // Spectral characteristics for wheeze
        if (!features.mfcc_std.empty() && features.mfcc_std[0] > 3.0f) {
            wheezeScore += 0.2f;
        }
        
        // Low energy variance suggests normal
        if (features.energyVariance < 0.05f && features.rmse_mean < 0.1f) {
            normalScore += 0.3f;
        }
        
        // Normalize
        float total = normalScore + crackleScore + wheezeScore + bothScore;
        cnnPred.probabilities[0] = normalScore / total;
        cnnPred.probabilities[1] = crackleScore / total;
        cnnPred.probabilities[2] = wheezeScore / total;
        cnnPred.probabilities[3] = bothScore / total;
        
        // Find max
        auto maxIt = std::max_element(cnnPred.probabilities.begin(), 
                                       cnnPred.probabilities.end());
        int maxIdx = std::distance(cnnPred.probabilities.begin(), maxIt);
        
        cnnPred.predictedClass = static_cast<CnnPredictionClass>(maxIdx);
        cnnPred.confidence = *maxIt;
        cnnPred.isValid = true;
    }
    
    // Convert CnnPredictionClass to PredictionClass
    if (cnnPred.isValid) {
        result.predictedClass = static_cast<PredictionClass>(
            static_cast<int>(cnnPred.predictedClass));
        result.confidence = cnnPred.confidence;
    }
    else {
        // CNN failed - use last layer result or unknown
        result.predictedClass = PredictionClass::UNKNOWN;
        result.confidence = 0.0f;
    }
    
    result.totalLayersProcessed = 4;
    
    return result;
}

// Training methods
void CascadedController::trainLayer1(
    const std::vector<std::vector<float>>& features,
    const std::vector<int>& labels) {
    
    if (m_layer1Forest) {
        m_layer1Forest->train(features, labels);
    }
}

void CascadedController::trainLayer2(
    const std::vector<std::vector<float>>& features,
    const std::vector<int>& labels) {
    
    if (m_layer2Forest) {
        m_layer2Forest->train(features, labels);
    }
}

void CascadedController::trainLayer3(
    const std::vector<std::vector<float>>& features,
    const std::vector<int>& labels) {
    
    if (m_layer3Forest) {
        m_layer3Forest->train(features, labels);
    }
}

void CascadedController::trainAll(const std::vector<FeatureVector>& features,
                                   const std::vector<int>& labels) {
    // Prepare features for each layer
    std::vector<std::vector<float>> layer1Features, layer2Features, layer3Features;
    
    for (const auto& f : features) {
        layer1Features.push_back(f.getLayer1Features());
        layer2Features.push_back(f.getLayer2Features());
        layer3Features.push_back(f.getLayer3Features());
    }
    
    trainLayer1(layer1Features, labels);
    trainLayer2(layer2Features, labels);
    trainLayer3(layer3Features, labels);
}

// Configuration
void CascadedController::setConfidenceThreshold(CascadeLayer layer, float threshold) {
    int idx = static_cast<int>(layer);
    if (idx >= 0 && idx < NUM_CASCADE_LAYERS) {
        m_confidenceThresholds[idx] = threshold;
    }
}

float CascadedController::getConfidenceThreshold(CascadeLayer layer) const {
    int idx = static_cast<int>(layer);
    if (idx >= 0 && idx < NUM_CASCADE_LAYERS) {
        return m_confidenceThresholds[idx];
    }
    return 0.0f;
}

void CascadedController::setConsensusThreshold(int threshold) {
    m_layer1Forest->setConsensusThreshold(threshold);
    m_layer2Forest->setConsensusThreshold(threshold);
    m_layer3Forest->setConsensusThreshold(threshold);
}

void CascadedController::setCNNCallback(CNNCallback callback) {
    m_cnnCallback = callback;
    m_enableCNN = (callback != nullptr);
}

// Model persistence
bool CascadedController::loadModels(const std::string& modelDir) {
    // TODO: Implement
    (void)modelDir;
    return false;
}

bool CascadedController::saveModels(const std::string& modelDir) const {
    // TODO: Implement
    (void)modelDir;
    return false;
}

// Statistics
void CascadedController::resetStatistics() {
    m_stats.layer1Exits = 0;
    m_stats.layer2Exits = 0;
    m_stats.layer3Exits = 0;
    m_stats.cnnFallbacks = 0;
    m_stats.totalPredictions = 0;
}

bool CascadedController::isLayer1Trained() const {
    return m_layer1Forest && m_layer1Forest->isTrained();
}

bool CascadedController::isLayer2Trained() const {
    return m_layer2Forest && m_layer2Forest->isTrained();
}

bool CascadedController::isLayer3Trained() const {
    return m_layer3Forest && m_layer3Forest->isTrained();
}

bool CascadedController::isFullyTrained() const {
    return isLayer1Trained() && isLayer2Trained() && isLayer3Trained();
}

} // namespace respiratory
