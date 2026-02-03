/**
 * @file CascadedLogic.h
 * @brief Cascaded Classification Framework for Respiratory Sound Analysis
 * 
 * Implements the 3-layer statistical screening framework from IEEE paper:
 * "Cascaded Framework with Hardware Acceleration for Respiratory Sound Analysis
 *  on Heterogeneous FPGA"
 * 
 * Architecture:
 *   Layer 1 (Global Screening): Metadata + Global Energy -> Quick healthy filter
 *   Layer 2 (Transient Detection): ZCR + Amplitude -> Crackle detection
 *   Layer 3 (Spectral Screening): MFCC -> Spectral anomaly detection
 *   Layer 4 (CNN): Deep learning for ambiguous cases [External]
 * 
 * Each layer uses Septuple Forest (7 RF clusters) with majority voting.
 * Early-exit mechanism when confidence threshold τ is exceeded.
 * 
 * @author Research Team
 * @date 2026
 */

#ifndef CASCADED_LOGIC_H
#define CASCADED_LOGIC_H

#include <vector>
#include <array>
#include <string>
#include <memory>
#include <functional>
#include <cstdint>
#include <map>

namespace respiratory {

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

class DecisionTree;
class RandomForest;
class SeptupleForest;
class CascadedController;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Số lượng cụm RF trong Septuple Forest
constexpr int NUM_FOREST_CLUSTERS = 7;

/// Ngưỡng đồng thuận tối thiểu (λ = 4)
constexpr int CONSENSUS_THRESHOLD = 4;

/// Số lượng layers trong cascade (không tính CNN)
constexpr int NUM_CASCADE_LAYERS = 3;

/// Số lượng classes
constexpr int NUM_CLASSES = 4;  // Normal, Crackle, Wheeze, Both

/// Số cây mặc định trong mỗi RF cluster
constexpr int DEFAULT_TREES_PER_CLUSTER = 10;

/// Độ sâu tối đa của cây quyết định
constexpr int MAX_TREE_DEPTH = 10;

/// Ngưỡng tin cậy mặc định cho early-exit
constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.7f;

// ============================================================================
// ENUMS
// ============================================================================

/**
 * @enum PredictionClass
 * @brief Các lớp phân loại
 */
enum class PredictionClass : int {
    NORMAL = 0,
    CRACKLE = 1,
    WHEEZE = 2,
    BOTH = 3,
    UNKNOWN = -1,       ///< Chưa xác định (cần layer tiếp theo)
    NEED_CNN = -2       ///< Cần kích hoạt CNN (Layer 4)
};

/**
 * @enum CascadeLayer
 * @brief Các tầng trong cascade
 */
enum class CascadeLayer : int {
    LAYER_1_GLOBAL = 0,     ///< Global Screening
    LAYER_2_TRANSIENT = 1,  ///< Transient Detection (Crackle)
    LAYER_3_SPECTRAL = 2,   ///< Spectral Screening (MFCC)
    LAYER_4_CNN = 3,        ///< CNN (external)
    COMPLETE = 4            ///< Đã hoàn thành phân loại
};

/**
 * @enum ExitReason
 * @brief Lý do thoát sớm
 */
enum class ExitReason {
    CONSENSUS_REACHED,      ///< Đạt ngưỡng đồng thuận (≥4/7)
    CONFIDENCE_EXCEEDED,    ///< Vượt ngưỡng tin cậy τ
    ALL_LAYERS_PASSED,      ///< Đã qua hết các layers
    NEED_DEEP_ANALYSIS      ///< Cần CNN để phân tích sâu
};

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @struct PatientMetadata
 * @brief Thông tin metadata bệnh nhân cho Layer 1
 */
struct PatientMetadata {
    int patientId;              ///< Mã bệnh nhân
    int age;                    ///< Tuổi (nếu có)
    bool hasPriorCondition;     ///< Có tiền sử bệnh hô hấp
    std::string chestLocation;  ///< Vị trí nghe
    std::string equipment;      ///< Thiết bị
    
    PatientMetadata() : patientId(0), age(0), hasPriorCondition(false) {}
};

/**
 * @struct FeatureVector
 * @brief Vector đặc trưng đầu vào cho classifier
 * 
 * Chứa tất cả features từ Phase 2:
 * - Time-domain: EED, ZCR, RMSE
 * - Frequency-domain: MFCC (39-dim)
 */
struct FeatureVector {
    // Metadata
    PatientMetadata metadata;
    
    // Time-domain features
    float eed;                          ///< Extreme Energy Difference
    float zcr_mean;                     ///< Mean Zero Crossing Rate
    float zcr_std;                      ///< Std Zero Crossing Rate
    float rmse_mean;                    ///< Mean RMSE
    float rmse_std;                     ///< Std RMSE
    
    // Global energy features (for Layer 1)
    float globalEnergy;                 ///< Năng lượng toàn cục
    float energyVariance;               ///< Phương sai năng lượng
    
    // MFCC features (for Layer 3)
    std::vector<float> mfcc_mean;       ///< Mean MFCC (13 dim)
    std::vector<float> mfcc_std;        ///< Std MFCC (13 dim)
    std::vector<float> delta_mean;      ///< Mean Delta (13 dim)
    std::vector<float> delta_std;       ///< Std Delta (13 dim)
    std::vector<float> delta2_mean;     ///< Mean Delta2 (13 dim)
    std::vector<float> delta2_std;      ///< Std Delta2 (13 dim)
    
    // Amplitude distribution features (for Layer 2)
    float amplitudeSkewness;            ///< Độ lệch phân phối biên độ
    float amplitudeKurtosis;            ///< Độ nhọn phân phối biên độ
    
    FeatureVector();
    
    /**
     * @brief Chuyển đổi thành flat vector
     * @return Vector phẳng tất cả features
     */
    std::vector<float> toFlatVector() const;
    
    /**
     * @brief Lấy features cho Layer 1 (Global Screening)
     */
    std::vector<float> getLayer1Features() const;
    
    /**
     * @brief Lấy features cho Layer 2 (Transient Detection)
     */
    std::vector<float> getLayer2Features() const;
    
    /**
     * @brief Lấy features cho Layer 3 (Spectral Screening)
     */
    std::vector<float> getLayer3Features() const;
};

/**
 * @struct ClassProbabilities
 * @brief Xác suất thuộc từng class
 */
struct ClassProbabilities {
    std::array<float, NUM_CLASSES> probs;  ///< Xác suất cho mỗi class
    
    ClassProbabilities() { probs.fill(0.0f); }
    
    /**
     * @brief Lấy class có xác suất cao nhất
     */
    PredictionClass getMaxClass() const;
    
    /**
     * @brief Lấy xác suất cao nhất (confidence)
     */
    float getMaxProbability() const;
    
    /**
     * @brief Chuẩn hóa xác suất về tổng = 1
     */
    void normalize();
};

/**
 * @struct ClusterVote
 * @brief Kết quả bỏ phiếu của một cluster
 */
struct ClusterVote {
    PredictionClass predictedClass;     ///< Class được dự đoán
    ClassProbabilities probabilities;   ///< Xác suất cho từng class
    float confidence;                   ///< Độ tin cậy
    
    ClusterVote() : predictedClass(PredictionClass::UNKNOWN), confidence(0.0f) {}
};

/**
 * @struct LayerResult
 * @brief Kết quả từ một layer
 */
struct LayerResult {
    CascadeLayer layer;                         ///< Layer đã xử lý
    std::array<ClusterVote, NUM_FOREST_CLUSTERS> clusterVotes;  ///< Phiếu từ 7 clusters
    std::array<int, NUM_CLASSES> voteCounts;    ///< Số phiếu cho mỗi class
    PredictionClass consensusClass;             ///< Class đạt đồng thuận (nếu có)
    float aggregatedConfidence;                 ///< Độ tin cậy tổng hợp
    bool hasConsensus;                          ///< Đã đạt ngưỡng đồng thuận?
    bool exceedsThreshold;                      ///< Vượt ngưỡng tin cậy τ?
    
    LayerResult() : layer(CascadeLayer::LAYER_1_GLOBAL),
                    consensusClass(PredictionClass::UNKNOWN),
                    aggregatedConfidence(0.0f),
                    hasConsensus(false),
                    exceedsThreshold(false) {
        voteCounts.fill(0);
    }
};

/**
 * @struct PredictionResult
 * @brief Kết quả phân loại cuối cùng
 */
struct PredictionResult {
    PredictionClass predictedClass;     ///< Class dự đoán cuối cùng
    float confidence;                   ///< Độ tin cậy
    CascadeLayer exitLayer;             ///< Layer đã thoát
    ExitReason exitReason;              ///< Lý do thoát
    std::vector<LayerResult> layerResults;  ///< Kết quả từ các layers đã qua
    int totalLayersProcessed;           ///< Số layers đã xử lý
    bool needsCNN;                      ///< Cần kích hoạt CNN?
    
    PredictionResult() : predictedClass(PredictionClass::UNKNOWN),
                         confidence(0.0f),
                         exitLayer(CascadeLayer::LAYER_1_GLOBAL),
                         exitReason(ExitReason::ALL_LAYERS_PASSED),
                         totalLayersProcessed(0),
                         needsCNN(false) {}
    
    /**
     * @brief Lấy tên class dạng string
     */
    std::string getClassName() const;
    
    /**
     * @brief Lấy mô tả kết quả
     */
    std::string describe() const;
};

// ============================================================================
// DECISION TREE CLASS
// ============================================================================

/**
 * @struct TreeNode
 * @brief Node trong cây quyết định
 */
struct TreeNode {
    bool isLeaf;                        ///< Là node lá?
    int featureIndex;                   ///< Index của feature để split (nếu không phải lá)
    float threshold;                    ///< Ngưỡng split
    int leftChild;                      ///< Index của node con trái
    int rightChild;                     ///< Index của node con phải
    ClassProbabilities leafProbs;       ///< Xác suất tại node lá
    
    TreeNode() : isLeaf(false), featureIndex(-1), threshold(0.0f),
                 leftChild(-1), rightChild(-1) {}
};

/**
 * @class DecisionTree
 * @brief Cây quyết định đơn lẻ
 * 
 * Thiết kế phù hợp với việc port sang FPGA:
 * - Sử dụng array-based tree storage
 * - Traversal đơn giản (if-else chain)
 * - Có thể export sang lookup table
 */
class DecisionTree {
public:
    DecisionTree();
    ~DecisionTree();
    
    /**
     * @brief Huấn luyện cây từ dữ liệu
     */
    void train(const std::vector<std::vector<float>>& features,
               const std::vector<int>& labels,
               int maxDepth = MAX_TREE_DEPTH);
    
    /**
     * @brief Dự đoán cho một sample
     */
    ClassProbabilities predict(const std::vector<float>& features) const;
    
    /**
     * @brief Load cây từ file/buffer (pre-trained)
     */
    bool loadFromBuffer(const std::vector<TreeNode>& nodes);
    
    /**
     * @brief Export cây ra buffer
     */
    std::vector<TreeNode> exportToBuffer() const;
    
    /**
     * @brief Kiểm tra cây đã được train chưa
     */
    bool isTrained() const { return m_isTrained; }

private:
    std::vector<TreeNode> m_nodes;      ///< Các nodes của cây
    int m_numFeatures;                  ///< Số features
    int m_maxDepth;                     ///< Độ sâu tối đa
    bool m_isTrained;                   ///< Đã train chưa
    
    /**
     * @brief Build node đệ quy
     */
    int buildNode(const std::vector<std::vector<float>>& features,
                  const std::vector<int>& labels,
                  const std::vector<int>& sampleIndices,
                  int depth);
    
    /**
     * @brief Tìm split tốt nhất cho một node
     */
    bool findBestSplit(const std::vector<std::vector<float>>& features,
                       const std::vector<int>& labels,
                       const std::vector<int>& sampleIndices,
                       int& bestFeature,
                       float& bestThreshold,
                       float& bestGini);
    
    /**
     * @brief Tính Gini impurity
     */
    float computeGini(const std::vector<int>& labels,
                      const std::vector<int>& indices) const;
};

// ============================================================================
// RANDOM FOREST CLASS
// ============================================================================

/**
 * @class RandomForest
 * @brief Ensemble của nhiều Decision Trees
 * 
 * Sử dụng bagging và random feature selection.
 */
class RandomForest {
public:
    /**
     * @brief Constructor
     * @param numTrees Số cây trong forest
     * @param maxDepth Độ sâu tối đa của mỗi cây
     */
    explicit RandomForest(int numTrees = DEFAULT_TREES_PER_CLUSTER,
                          int maxDepth = MAX_TREE_DEPTH);
    ~RandomForest();
    
    /**
     * @brief Huấn luyện forest
     */
    void train(const std::vector<std::vector<float>>& features,
               const std::vector<int>& labels);
    
    /**
     * @brief Dự đoán cho một sample
     */
    ClassProbabilities predict(const std::vector<float>& features) const;
    
    /**
     * @brief Dự đoán và trả về class
     */
    PredictionClass predictClass(const std::vector<float>& features) const;
    
    /**
     * @brief Load pre-trained model
     */
    bool loadModel(const std::string& modelPath);
    
    /**
     * @brief Save model
     */
    bool saveModel(const std::string& modelPath) const;
    
    /**
     * @brief Kiểm tra đã train chưa
     */
    bool isTrained() const { return m_isTrained; }
    
    /**
     * @brief Lấy số cây
     */
    int getNumTrees() const { return m_numTrees; }

private:
    int m_numTrees;
    int m_maxDepth;
    std::vector<std::unique_ptr<DecisionTree>> m_trees;
    bool m_isTrained;
    
    /**
     * @brief Tạo bootstrap sample
     */
    void createBootstrapSample(const std::vector<std::vector<float>>& features,
                               const std::vector<int>& labels,
                               std::vector<std::vector<float>>& bootFeatures,
                               std::vector<int>& bootLabels);
};

// ============================================================================
// SEPTUPLE FOREST CLASS
// ============================================================================

/**
 * @class SeptupleForest
 * @brief 7 cụm Random Forest hoạt động song song với majority voting
 * 
 * Theo bài báo IEEE:
 * - 7 RF clusters chạy song song
 * - Majority voting với ngưỡng λ = 4
 * - Early exit khi đạt đồng thuận
 * 
 * Công thức:
 * Decision = Class c  nếu Σ I(Oj = c) ≥ 4
 *          = Next Layer  nếu ngược lại
 */
class SeptupleForest {
public:
    /**
     * @brief Constructor
     * @param treesPerCluster Số cây trong mỗi cluster
     * @param consensusThreshold Ngưỡng đồng thuận (mặc định 4)
     */
    explicit SeptupleForest(int treesPerCluster = DEFAULT_TREES_PER_CLUSTER,
                            int consensusThreshold = CONSENSUS_THRESHOLD);
    ~SeptupleForest();
    
    /**
     * @brief Huấn luyện tất cả 7 clusters
     */
    void train(const std::vector<std::vector<float>>& features,
               const std::vector<int>& labels);
    
    /**
     * @brief Dự đoán với majority voting
     * 
     * @param features Vector đặc trưng
     * @param result Kết quả chi tiết (output)
     * @return true nếu đạt đồng thuận, false nếu cần layer tiếp theo
     */
    bool predictWithVoting(const std::vector<float>& features,
                           LayerResult& result) const;
    
    /**
     * @brief Dự đoán đơn giản (không chi tiết)
     */
    ClusterVote predict(const std::vector<float>& features) const;
    
    /**
     * @brief Load pre-trained models cho tất cả clusters
     */
    bool loadModels(const std::string& basePath);
    
    /**
     * @brief Save models
     */
    bool saveModels(const std::string& basePath) const;
    
    /**
     * @brief Đặt ngưỡng đồng thuận
     */
    void setConsensusThreshold(int threshold) { m_consensusThreshold = threshold; }
    
    /**
     * @brief Lấy ngưỡng đồng thuận
     */
    int getConsensusThreshold() const { return m_consensusThreshold; }
    
    /**
     * @brief Kiểm tra đã train chưa
     */
    bool isTrained() const;

private:
    std::array<std::unique_ptr<RandomForest>, NUM_FOREST_CLUSTERS> m_clusters;
    int m_treesPerCluster;
    int m_consensusThreshold;
    
    /**
     * @brief Tính majority vote từ kết quả các clusters
     */
    void computeMajorityVote(const std::array<ClusterVote, NUM_FOREST_CLUSTERS>& votes,
                             LayerResult& result) const;
};

// ============================================================================
// CASCADED CONTROLLER CLASS
// ============================================================================

/**
 * @class CascadedController
 * @brief Điều khiển luồng phân loại phân tầng
 * 
 * Quản lý 3 layers:
 * - Layer 1: Global Screening (Healthy filter)
 * - Layer 2: Transient Detection (Crackle)
 * - Layer 3: Spectral Screening (MFCC-based)
 * 
 * Early-exit mechanism:
 * - Thoát khi đạt đồng thuận (≥4/7 clusters)
 * - Thoát khi vượt ngưỡng tin cậy τ
 * - Chuyển sang CNN nếu ambiguous
 * 
 * Thiết kế hướng FPGA:
 * - State machine pattern
 * - Dễ dàng thay thế RF bằng IP Core driver
 * - Interface rõ ràng cho hardware acceleration
 */
class CascadedController {
public:
    /**
     * @brief Callback type cho CNN inference (external)
     */
    using CNNCallback = std::function<PredictionClass(const FeatureVector&)>;
    
    /**
     * @brief Constructor
     */
    CascadedController();
    
    /**
     * @brief Destructor
     */
    ~CascadedController();
    
    // ========================================================================
    // MAIN PREDICTION INTERFACE
    // ========================================================================
    
    /**
     * @brief Dự đoán class cho một sample
     * 
     * Điều phối luồng dữ liệu qua các layers:
     * 1. Layer 1 (Global) -> Early exit nếu healthy
     * 2. Layer 2 (Transient) -> Early exit nếu crackle
     * 3. Layer 3 (Spectral) -> Early exit nếu đủ tin cậy
     * 4. Signal CNN nếu ambiguous
     * 
     * @param features Vector đặc trưng đầu vào
     * @return Kết quả phân loại chi tiết
     */
    PredictionResult predict(const FeatureVector& features);
    
    /**
     * @brief Dự đoán batch nhiều samples
     */
    std::vector<PredictionResult> predictBatch(
        const std::vector<FeatureVector>& features);
    
    // ========================================================================
    // TRAINING INTERFACE
    // ========================================================================
    
    /**
     * @brief Huấn luyện Layer 1 (Global Screening)
     */
    void trainLayer1(const std::vector<std::vector<float>>& features,
                     const std::vector<int>& labels);
    
    /**
     * @brief Huấn luyện Layer 2 (Transient Detection)
     */
    void trainLayer2(const std::vector<std::vector<float>>& features,
                     const std::vector<int>& labels);
    
    /**
     * @brief Huấn luyện Layer 3 (Spectral Screening)
     */
    void trainLayer3(const std::vector<std::vector<float>>& features,
                     const std::vector<int>& labels);
    
    /**
     * @brief Huấn luyện tất cả layers
     */
    void trainAll(const std::vector<FeatureVector>& features,
                  const std::vector<int>& labels);
    
    // ========================================================================
    // CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Đặt ngưỡng tin cậy cho từng layer
     * @param layer Layer cần đặt
     * @param threshold Ngưỡng (0.0 - 1.0)
     */
    void setConfidenceThreshold(CascadeLayer layer, float threshold);
    
    /**
     * @brief Lấy ngưỡng tin cậy
     */
    float getConfidenceThreshold(CascadeLayer layer) const;
    
    /**
     * @brief Đặt ngưỡng đồng thuận cho tất cả layers
     */
    void setConsensusThreshold(int threshold);
    
    /**
     * @brief Đăng ký callback cho CNN inference
     */
    void setCNNCallback(CNNCallback callback);
    
    /**
     * @brief Bật/tắt CNN fallback
     */
    void enableCNNFallback(bool enable) { m_enableCNN = enable; }
    
    // ========================================================================
    // MODEL PERSISTENCE
    // ========================================================================
    
    /**
     * @brief Load tất cả models từ thư mục
     */
    bool loadModels(const std::string& modelDir);
    
    /**
     * @brief Save tất cả models
     */
    bool saveModels(const std::string& modelDir) const;
    
    // ========================================================================
    // STATISTICS & DEBUGGING
    // ========================================================================
    
    /**
     * @brief Lấy thống kê early-exit
     */
    struct ExitStatistics {
        int layer1Exits;
        int layer2Exits;
        int layer3Exits;
        int cnnFallbacks;
        int totalPredictions;
        
        float getLayer1ExitRate() const {
            return totalPredictions > 0 ? 
                   static_cast<float>(layer1Exits) / totalPredictions : 0.0f;
        }
    };
    
    ExitStatistics getExitStatistics() const { return m_stats; }
    
    /**
     * @brief Reset thống kê
     */
    void resetStatistics();
    
    /**
     * @brief Kiểm tra layers đã được train chưa
     */
    bool isLayer1Trained() const;
    bool isLayer2Trained() const;
    bool isLayer3Trained() const;
    bool isFullyTrained() const;

private:
    // ========================================================================
    // PRIVATE MEMBERS
    // ========================================================================
    
    /// Septuple Forests cho 3 layers
    std::unique_ptr<SeptupleForest> m_layer1Forest;  ///< Global Screening
    std::unique_ptr<SeptupleForest> m_layer2Forest;  ///< Transient Detection
    std::unique_ptr<SeptupleForest> m_layer3Forest;  ///< Spectral Screening
    
    /// Ngưỡng tin cậy cho từng layer
    std::array<float, NUM_CASCADE_LAYERS> m_confidenceThresholds;
    
    /// CNN callback (optional)
    CNNCallback m_cnnCallback;
    bool m_enableCNN;
    
    /// Thống kê
    mutable ExitStatistics m_stats;
    
    // ========================================================================
    // PRIVATE METHODS
    // ========================================================================
    
    /**
     * @brief Xử lý Layer 1 - Global Screening
     * 
     * Sử dụng metadata và global energy để lọc nhanh healthy cases.
     */
    LayerResult processLayer1(const FeatureVector& features);
    
    /**
     * @brief Xử lý Layer 2 - Transient Detection
     * 
     * Phân tích ZCR và amplitude distribution để phát hiện Crackles.
     */
    LayerResult processLayer2(const FeatureVector& features);
    
    /**
     * @brief Xử lý Layer 3 - Spectral Screening
     * 
     * Sử dụng MFCC để phát hiện bất thường phổ âm.
     */
    LayerResult processLayer3(const FeatureVector& features);
    
    /**
     * @brief Kiểm tra điều kiện early-exit
     */
    bool checkEarlyExit(const LayerResult& result, CascadeLayer layer) const;
    
    /**
     * @brief Gọi CNN fallback (nếu có)
     */
    PredictionClass invokeCNN(const FeatureVector& features);
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Chuyển PredictionClass sang string
 */
std::string predictionClassToString(PredictionClass cls);

/**
 * @brief Chuyển CascadeLayer sang string
 */
std::string cascadeLayerToString(CascadeLayer layer);

/**
 * @brief Chuyển ExitReason sang string
 */
std::string exitReasonToString(ExitReason reason);

/**
 * @brief Tính accuracy từ predictions và labels
 */
float computeAccuracy(const std::vector<PredictionClass>& predictions,
                      const std::vector<int>& labels);

/**
 * @brief In confusion matrix
 */
void printConfusionMatrix(const std::vector<PredictionClass>& predictions,
                          const std::vector<int>& labels);

} // namespace respiratory

// ============================================================================
// TRAINING & EVALUATION UTILITIES (requires full type definitions)
// These are declared after namespace to allow forward reference resolution
// ============================================================================

// Include here to get full type definitions
#include "FeatureExtraction.h"
#include "SignalPrep.hpp"

namespace respiratory {

/**
 * @brief Chuyển đổi CycleFeatures sang FeatureVector cho classifier
 */
FeatureVector cycleFeaturesToFeatureVector(const CycleFeatures& cycleFeatures,
                                            const PatientMetadata& metadata);

/**
 * @brief Train và evaluate Cascaded Controller
 */
void trainAndEvaluateCascaded(
    const std::vector<LabeledBreathingCycle>& trainData,
    const std::vector<LabeledBreathingCycle>& testData,
    CascadedController& controller,
    bool verbose);

} // namespace respiratory

#endif // CASCADED_LOGIC_H
