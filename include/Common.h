/**
 * @file Common.h
 * @brief Common definitions and types for Respiratory Sound Analysis
 * 
 * Chứa các định nghĩa chung, types, và constants được sử dụng 
 * xuyên suốt dự án.
 */

#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <vector>
#include <string>
#include <cmath>

namespace respiratory {

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/// Kiểu dữ liệu cho samples (có thể thay đổi cho fixed-point HLS)
using SampleType = float;

/// Kiểu dữ liệu cho accumulator (precision cao hơn cho tính toán)
using AccumType = double;

/// Kiểu dữ liệu cho index
using IndexType = uint32_t;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Pi constant
constexpr double PI = 3.14159265358979323846;

/// Epsilon để tránh chia cho 0
constexpr float EPSILON = 1e-10f;

// ============================================================================
// ENUMS
// ============================================================================

/**
 * @enum DiagnosisResult
 * @brief Kết quả chẩn đoán theo Cascaded Framework
 */
enum class DiagnosisResult {
    NORMAL = 0,           ///< Bình thường
    ABNORMAL = 1,         ///< Bất thường (chưa xác định loại)
    CRACKLE = 2,          ///< Tiếng rít (Crackle)
    WHEEZE = 3,           ///< Tiếng khò khè (Wheeze)
    CRACKLE_WHEEZE = 4,   ///< Cả hai loại
    UNKNOWN = 5           ///< Không xác định được
};

/**
 * @enum ProcessingStage
 * @brief Các giai đoạn xử lý trong pipeline
 */
enum class ProcessingStage {
    LOADING = 0,          ///< Đang tải file
    RESAMPLING = 1,       ///< Đang resampling
    FILTERING = 2,        ///< Đang lọc
    NORMALIZING = 3,      ///< Đang chuẩn hóa
    SEGMENTING = 4,       ///< Đang phân đoạn
    EXTRACTING = 5,       ///< Đang trích xuất features
    CLASSIFYING = 6,      ///< Đang phân loại
    COMPLETE = 7          ///< Hoàn thành
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Chuyển đổi DiagnosisResult sang string
 */
inline std::string diagnosisToString(DiagnosisResult result) {
    switch (result) {
        case DiagnosisResult::NORMAL: return "Normal";
        case DiagnosisResult::ABNORMAL: return "Abnormal";
        case DiagnosisResult::CRACKLE: return "Crackle";
        case DiagnosisResult::WHEEZE: return "Wheeze";
        case DiagnosisResult::CRACKLE_WHEEZE: return "Crackle+Wheeze";
        default: return "Unknown";
    }
}

/**
 * @brief Chuyển đổi ProcessingStage sang string
 */
inline std::string stageToString(ProcessingStage stage) {
    switch (stage) {
        case ProcessingStage::LOADING: return "Loading";
        case ProcessingStage::RESAMPLING: return "Resampling";
        case ProcessingStage::FILTERING: return "Filtering";
        case ProcessingStage::NORMALIZING: return "Normalizing";
        case ProcessingStage::SEGMENTING: return "Segmenting";
        case ProcessingStage::EXTRACTING: return "Feature Extraction";
        case ProcessingStage::CLASSIFYING: return "Classification";
        case ProcessingStage::COMPLETE: return "Complete";
        default: return "Unknown Stage";
    }
}

} // namespace respiratory

#endif // COMMON_H
