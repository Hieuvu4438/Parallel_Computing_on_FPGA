/**
 * @file SignalPrep.hpp
 * @brief Signal Preprocessing Module for Respiratory Sound Analysis
 * 
 * Implements the preprocessing pipeline from IEEE paper:
 * "Cascaded Framework with Hardware Acceleration for Respiratory Sound Analysis
 *  on Heterogeneous FPGA"
 * 
 * Pipeline stages:
 *   1. WAV file loading (via dr_wav)
 *   2. Resampling to 4kHz target frequency
 *   3. Band-pass filtering (50Hz - 2500Hz)
 *   4. Amplitude normalization to [-1.0, 1.0]
 *   5. Breathing cycle segmentation (based on ICBHI annotations)
 * 
 * ICBHI 2017 Dataset Integration:
 *   - Automatic dataset scanning
 *   - Annotation file parsing
 *   - Patient metadata extraction from filenames
 * 
 * @author Research Team
 * @date 2026
 */

#ifndef SIGNAL_PREP_HPP
#define SIGNAL_PREP_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <cmath>
#include <map>
#include <functional>

// Define M_PI if not available (Windows/MinGW compatibility)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace respiratory {

// ============================================================================
// CONSTANTS - Theo thông số kỹ thuật từ bài báo IEEE
// ============================================================================

/// Tần số lấy mẫu mục tiêu (Hz) - Target sampling rate
constexpr float TARGET_SAMPLE_RATE = 4000.0f;

/// Tần số cắt thấp của bộ lọc band-pass (Hz) - Low cutoff frequency
constexpr float BANDPASS_LOW_FREQ = 50.0f;

/// Tần số cắt cao của bộ lọc band-pass (Hz) - High cutoff frequency  
constexpr float BANDPASS_HIGH_FREQ = 2500.0f;

/// Bậc của bộ lọc Butterworth - Filter order
constexpr int FILTER_ORDER = 4;

/// Ngưỡng năng lượng tối thiểu để phát hiện chu kỳ hô hấp
/// Minimum energy threshold for breathing cycle detection
constexpr float ENERGY_THRESHOLD_RATIO = 0.1f;

/// Độ dài tối thiểu của một chu kỳ hô hấp (samples tại 4kHz)
/// Minimum breathing cycle length (~0.5 seconds)
constexpr int MIN_CYCLE_LENGTH = 2000;

/// Độ dài tối đa của một chu kỳ hô hấp (samples tại 4kHz)
/// Maximum breathing cycle length (~8 seconds)
constexpr int MAX_CYCLE_LENGTH = 32000;

/// Kích thước cửa sổ để tính năng lượng (samples)
/// Window size for energy calculation
constexpr int ENERGY_WINDOW_SIZE = 200;

/// Đường dẫn mặc định đến thư mục ICBHI dataset
/// Default path to ICBHI dataset directory
const std::string DEFAULT_DATASET_PATH = "data/samples/ICBHI_final_database";

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @struct AudioData
 * @brief Cấu trúc lưu trữ dữ liệu âm thanh và metadata
 */
struct AudioData {
    std::vector<float> samples;     ///< Mảng mẫu tín hiệu (normalized float)
    uint32_t sampleRate;            ///< Tần số lấy mẫu gốc (Hz)
    uint16_t channels;              ///< Số kênh âm thanh
    uint32_t totalSamples;          ///< Tổng số mẫu
    
    AudioData() : sampleRate(0), channels(0), totalSamples(0) {}
};

/**
 * @struct BreathingCycle
 * @brief Cấu trúc lưu trữ một chu kỳ hô hấp đã được phân đoạn
 */
struct BreathingCycle {
    std::vector<float> samples;     ///< Dữ liệu mẫu của chu kỳ
    size_t startIndex;              ///< Vị trí bắt đầu trong tín hiệu gốc
    size_t endIndex;                ///< Vị trí kết thúc trong tín hiệu gốc
    float duration;                 ///< Thời lượng (giây)
    float averageEnergy;            ///< Năng lượng trung bình
    
    BreathingCycle() : startIndex(0), endIndex(0), duration(0.0f), averageEnergy(0.0f) {}
};

// ============================================================================
// ICBHI DATASET STRUCTURES - Cấu trúc dữ liệu cho ICBHI 2017 Dataset
// ============================================================================

/**
 * @enum RespiratoryLabel
 * @brief Nhãn phân loại âm thanh hô hấp theo ICBHI
 * 
 * Hệ thống nhãn 4 lớp theo bài báo IEEE:
 * - Normal: Không có crackle, không có wheeze
 * - Crackle: Chỉ có crackle
 * - Wheeze: Chỉ có wheeze
 * - Both: Có cả crackle và wheeze
 */
enum class RespiratoryLabel {
    NORMAL = 0,         ///< Bình thường (Crackle=0, Wheeze=0)
    CRACKLE = 1,        ///< Có tiếng rít (Crackle=1, Wheeze=0)
    WHEEZE = 2,         ///< Có tiếng khò khè (Crackle=0, Wheeze=1)
    BOTH = 3            ///< Cả hai loại (Crackle=1, Wheeze=1)
};

/**
 * @struct CycleAnnotation
 * @brief Annotation cho một chu kỳ hô hấp từ file .txt ICBHI
 * 
 * Format file ICBHI: [Start] [End] [Crackles] [Wheezes]
 * Ví dụ: 0.036  0.579  0  0
 */
struct CycleAnnotation {
    float startTime;        ///< Thời điểm bắt đầu (giây)
    float endTime;          ///< Thời điểm kết thúc (giây)
    bool hasCrackle;        ///< Có tiếng crackle (1) hay không (0)
    bool hasWheeze;         ///< Có tiếng wheeze (1) hay không (0)
    
    CycleAnnotation() 
        : startTime(0.0f), endTime(0.0f), hasCrackle(false), hasWheeze(false) {}
    
    /**
     * @brief Lấy nhãn phân loại từ annotation
     * @return RespiratoryLabel tương ứng
     */
    RespiratoryLabel getLabel() const {
        if (hasCrackle && hasWheeze) return RespiratoryLabel::BOTH;
        if (hasCrackle) return RespiratoryLabel::CRACKLE;
        if (hasWheeze) return RespiratoryLabel::WHEEZE;
        return RespiratoryLabel::NORMAL;
    }
    
    /**
     * @brief Tính thời lượng của chu kỳ
     * @return Thời lượng (giây)
     */
    float getDuration() const {
        return endTime - startTime;
    }
};

/**
 * @struct PatientInfo
 * @brief Thông tin bệnh nhân trích xuất từ tên file ICBHI
 * 
 * Format tên file: {PatientID}_{RecordingIndex}_{ChestLocation}_{Mode}_{Equipment}.wav
 * Ví dụ: 101_1b1_Al_sc_Meditron.wav
 *        101 = Patient ID
 *        1b1 = Recording index
 *        Al = Anterior left
 *        sc = Single channel
 *        Meditron = Stethoscope model
 */
struct PatientInfo {
    int patientId;              ///< Mã bệnh nhân (101-226)
    std::string recordingIndex; ///< Chỉ số bản ghi (1b1, 2b2, etc.)
    std::string chestLocation;  ///< Vị trí nghe (Al, Ar, Pl, Pr, Tc, etc.)
    std::string acquisitionMode;///< Chế độ thu (sc=single, mc=multi channel)
    std::string equipment;      ///< Thiết bị (Meditron, LittC2SE, etc.)
    
    PatientInfo() : patientId(0) {}
};

/**
 * @struct LabeledBreathingCycle
 * @brief Chu kỳ hô hấp đã được gán nhãn - Đầu ra chính của DatasetManager
 * 
 * Cấu trúc này kết hợp:
 * - Dữ liệu tín hiệu đã xử lý (resampled, filtered, normalized)
 * - Nhãn phân loại từ annotation
 * - Thông tin bệnh nhân để phân tích
 */
struct LabeledBreathingCycle {
    std::vector<float> samples;     ///< Dữ liệu tín hiệu đã tiền xử lý
    RespiratoryLabel label;         ///< Nhãn phân loại (Normal/Crackle/Wheeze/Both)
    PatientInfo patientInfo;        ///< Thông tin bệnh nhân
    float startTime;                ///< Thời điểm bắt đầu trong file gốc (giây)
    float endTime;                  ///< Thời điểm kết thúc trong file gốc (giây)
    float duration;                 ///< Thời lượng (giây)
    std::string sourceFile;         ///< Tên file nguồn (để truy vết)
    
    LabeledBreathingCycle() 
        : label(RespiratoryLabel::NORMAL)
        , startTime(0.0f), endTime(0.0f), duration(0.0f) {}
    
    /**
     * @brief Chuyển nhãn thành chuỗi để hiển thị
     */
    std::string getLabelString() const {
        switch (label) {
            case RespiratoryLabel::NORMAL: return "Normal";
            case RespiratoryLabel::CRACKLE: return "Crackle";
            case RespiratoryLabel::WHEEZE: return "Wheeze";
            case RespiratoryLabel::BOTH: return "Both";
            default: return "Unknown";
        }
    }
};

/**
 * @struct DatasetStatistics
 * @brief Thống kê về dataset đã load
 */
struct DatasetStatistics {
    size_t totalFiles;              ///< Tổng số file WAV
    size_t totalCycles;             ///< Tổng số chu kỳ hô hấp
    size_t normalCount;             ///< Số chu kỳ Normal
    size_t crackleCount;            ///< Số chu kỳ Crackle
    size_t wheezeCount;             ///< Số chu kỳ Wheeze
    size_t bothCount;               ///< Số chu kỳ Both
    size_t uniquePatients;          ///< Số bệnh nhân unique
    float avgCycleDuration;         ///< Thời lượng trung bình (giây)
    
    DatasetStatistics() 
        : totalFiles(0), totalCycles(0)
        , normalCount(0), crackleCount(0), wheezeCount(0), bothCount(0)
        , uniquePatients(0), avgCycleDuration(0.0f) {}
    
    /**
     * @brief In thống kê ra console
     */
    void print() const;
};

/**
 * @struct FilterCoefficients
 * @brief Hệ số bộ lọc IIR (Butterworth)
 * 
 * Bộ lọc dạng: y[n] = b0*x[n] + b1*x[n-1] + ... - a1*y[n-1] - a2*y[n-2] - ...
 */
struct FilterCoefficients {
    std::vector<double> b;          ///< Hệ số tử số (feedforward)
    std::vector<double> a;          ///< Hệ số mẫu số (feedback)
    
    FilterCoefficients() = default;
};

// ============================================================================
// MAIN CLASS
// ============================================================================

/**
 * @class SignalProcessor
 * @brief Lớp xử lý tín hiệu chính cho phân tích âm thanh hô hấp
 * 
 * Lớp này thực hiện toàn bộ pipeline tiền xử lý theo bài báo IEEE:
 * - Đọc file WAV
 * - Resampling về 4kHz
 * - Lọc band-pass 50-2500Hz
 * - Chuẩn hóa biên độ
 * - Phân đoạn chu kỳ hô hấp
 * 
 * Thiết kế hướng tới việc chuyển đổi sang Vitis HLS sau này:
 * - Tránh cấp phát động trong các vòng lặp chính
 * - Sử dụng các phép toán thân thiện với fixed-point
 * - Comment rõ ràng cho từng bước xử lý
 */
class SignalProcessor {
public:
    // ========================================================================
    // CONSTRUCTORS & DESTRUCTOR
    // ========================================================================
    
    /**
     * @brief Constructor mặc định
     */
    SignalProcessor();
    
    /**
     * @brief Destructor
     */
    ~SignalProcessor();
    
    // ========================================================================
    // MAIN PIPELINE METHODS
    // ========================================================================
    
    /**
     * @brief Thực hiện toàn bộ pipeline tiền xử lý
     * 
     * @param inputPath Đường dẫn file WAV đầu vào
     * @param outputCycles Vector chứa các chu kỳ hô hấp đã phân đoạn
     * @return true nếu xử lý thành công
     */
    bool processFile(const std::string& inputPath, 
                     std::vector<BreathingCycle>& outputCycles);
    
    /**
     * @brief Xử lý dữ liệu âm thanh đã có trong bộ nhớ
     * 
     * @param inputSamples Vector mẫu đầu vào
     * @param inputSampleRate Tần số lấy mẫu đầu vào
     * @param outputCycles Vector chứa các chu kỳ hô hấp đã phân đoạn
     * @return true nếu xử lý thành công
     */
    bool processBuffer(const std::vector<float>& inputSamples,
                       uint32_t inputSampleRate,
                       std::vector<BreathingCycle>& outputCycles);
    
    // ========================================================================
    // INDIVIDUAL PROCESSING STAGES
    // ========================================================================
    
    /**
     * @brief Đọc file WAV và trả về dữ liệu âm thanh
     * 
     * Sử dụng thư viện dr_wav để đọc file WAV.
     * Hỗ trợ các định dạng: 8-bit, 16-bit, 24-bit, 32-bit, float.
     * 
     * @param filePath Đường dẫn file WAV
     * @param audioData Cấu trúc lưu trữ dữ liệu đọc được
     * @return true nếu đọc thành công
     */
    bool loadWavFile(const std::string& filePath, AudioData& audioData);
    
    /**
     * @brief Resampling tín hiệu về tần số mục tiêu (4kHz)
     * 
     * Sử dụng phương pháp nội suy tuyến tính (linear interpolation).
     * Phương pháp này đơn giản và phù hợp với HLS implementation.
     * 
     * @param input Vector mẫu đầu vào
     * @param inputRate Tần số lấy mẫu đầu vào (Hz)
     * @param output Vector mẫu đầu ra (đã resampled)
     * @param targetRate Tần số lấy mẫu mục tiêu (Hz), mặc định 4kHz
     */
    void resample(const std::vector<float>& input, 
                  uint32_t inputRate,
                  std::vector<float>& output,
                  float targetRate = TARGET_SAMPLE_RATE);
    
    /**
     * @brief Áp dụng bộ lọc band-pass Butterworth
     * 
     * Lọc tín hiệu trong dải 50Hz - 2500Hz theo bài báo.
     * Sử dụng bộ lọc Butterworth bậc 4 để có đáp ứng phẳng trong dải thông.
     * Áp dụng lọc zero-phase (forward-backward) để tránh méo pha.
     * 
     * @param input Vector mẫu đầu vào
     * @param output Vector mẫu đầu ra (đã lọc)
     * @param sampleRate Tần số lấy mẫu của tín hiệu (Hz)
     * @param lowCutoff Tần số cắt thấp (Hz)
     * @param highCutoff Tần số cắt cao (Hz)
     */
    void applyBandpassFilter(const std::vector<float>& input,
                             std::vector<float>& output,
                             float sampleRate = TARGET_SAMPLE_RATE,
                             float lowCutoff = BANDPASS_LOW_FREQ,
                             float highCutoff = BANDPASS_HIGH_FREQ);
    
    /**
     * @brief Chuẩn hóa tín hiệu về khoảng [-1.0, 1.0]
     * 
     * Sử dụng phương pháp peak normalization:
     * normalized[i] = samples[i] / max(|samples|)
     * 
     * @param samples Vector mẫu (được chuẩn hóa tại chỗ)
     */
    void normalize(std::vector<float>& samples);
    
    /**
     * @brief Phân đoạn tín hiệu thành các chu kỳ hô hấp
     * 
     * Thuật toán:
     * 1. Tính năng lượng ngắn hạn (short-time energy)
     * 2. Tìm các điểm bắt đầu/kết thúc dựa trên ngưỡng năng lượng
     * 3. Áp dụng ràng buộc về độ dài chu kỳ (0.5s - 8s)
     * 4. Merge các segment quá ngắn
     * 
     * @param samples Vector mẫu đầu vào (đã preprocessed)
     * @param cycles Vector chứa các chu kỳ hô hấp phát hiện được
     * @param sampleRate Tần số lấy mẫu (Hz)
     */
    void segmentBreathingCycles(const std::vector<float>& samples,
                                std::vector<BreathingCycle>& cycles,
                                float sampleRate = TARGET_SAMPLE_RATE);
    
    // ========================================================================
    // UTILITY METHODS
    // ========================================================================
    
    /**
     * @brief Lấy tín hiệu đã xử lý hoàn chỉnh
     * @return Reference đến vector mẫu đã xử lý
     */
    const std::vector<float>& getProcessedSignal() const;
    
    /**
     * @brief Lấy tần số lấy mẫu hiện tại
     * @return Tần số lấy mẫu (Hz)
     */
    float getCurrentSampleRate() const;
    
    /**
     * @brief Reset trạng thái processor
     */
    void reset();
    
    /**
     * @brief Kiểm tra xem processor đã có dữ liệu hợp lệ chưa
     * @return true nếu có dữ liệu hợp lệ
     */
    bool hasValidData() const;

private:
    // ========================================================================
    // PRIVATE HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Thiết kế bộ lọc Butterworth lowpass
     * 
     * Tính toán hệ số bộ lọc Butterworth lowpass bậc n.
     * Sử dụng biến đổi bilinear để chuyển từ analog sang digital.
     * 
     * @param cutoffFreq Tần số cắt (Hz)
     * @param sampleRate Tần số lấy mẫu (Hz)
     * @param order Bậc của bộ lọc
     * @param coeffs Cấu trúc lưu hệ số bộ lọc
     */
    void designButterworthLowpass(float cutoffFreq, 
                                   float sampleRate,
                                   int order,
                                   FilterCoefficients& coeffs);
    
    /**
     * @brief Thiết kế bộ lọc Butterworth highpass
     * 
     * @param cutoffFreq Tần số cắt (Hz)
     * @param sampleRate Tần số lấy mẫu (Hz)
     * @param order Bậc của bộ lọc
     * @param coeffs Cấu trúc lưu hệ số bộ lọc
     */
    void designButterworthHighpass(float cutoffFreq,
                                    float sampleRate,
                                    int order,
                                    FilterCoefficients& coeffs);
    
    /**
     * @brief Áp dụng bộ lọc IIR cho tín hiệu
     * 
     * Thực hiện lọc IIR dạng Direct Form II Transposed.
     * Công thức: y[n] = b0*x[n] + b1*x[n-1] + ... - a1*y[n-1] - ...
     * 
     * @param input Vector mẫu đầu vào
     * @param output Vector mẫu đầu ra
     * @param coeffs Hệ số bộ lọc
     */
    void applyIIRFilter(const std::vector<float>& input,
                        std::vector<float>& output,
                        const FilterCoefficients& coeffs);
    
    /**
     * @brief Áp dụng lọc zero-phase (forward-backward filtering)
     * 
     * Lọc tín hiệu theo chiều xuôi, sau đó đảo ngược và lọc lại.
     * Kết quả là bộ lọc có pha bằng 0 (không có độ trễ nhóm).
     * 
     * @param input Vector mẫu đầu vào
     * @param output Vector mẫu đầu ra
     * @param coeffs Hệ số bộ lọc
     */
    void applyZeroPhaseFilter(const std::vector<float>& input,
                              std::vector<float>& output,
                              const FilterCoefficients& coeffs);
    
    /**
     * @brief Tính năng lượng ngắn hạn của tín hiệu
     * 
     * E[n] = sum(x[n-w/2:n+w/2]^2) / w
     * 
     * @param samples Vector mẫu
     * @param energy Vector năng lượng đầu ra
     * @param windowSize Kích thước cửa sổ
     */
    void computeShortTimeEnergy(const std::vector<float>& samples,
                                std::vector<float>& energy,
                                int windowSize = ENERGY_WINDOW_SIZE);
    
    /**
     * @brief Chuyển đổi tín hiệu stereo sang mono
     * 
     * mono[i] = (left[i] + right[i]) / 2
     * 
     * @param input Vector mẫu stereo interleaved
     * @param output Vector mẫu mono
     * @param channels Số kênh
     */
    void convertToMono(const std::vector<float>& input,
                       std::vector<float>& output,
                       uint16_t channels);

    // ========================================================================
    // MEMBER VARIABLES
    // ========================================================================
    
    std::vector<float> m_processedSignal;   ///< Tín hiệu đã xử lý
    float m_currentSampleRate;               ///< Tần số lấy mẫu hiện tại
    bool m_hasValidData;                     ///< Cờ trạng thái dữ liệu
    
    /// Hệ số bộ lọc lowpass (được tính trước để tái sử dụng)
    FilterCoefficients m_lowpassCoeffs;
    
    /// Hệ số bộ lọc highpass (được tính trước để tái sử dụng)
    FilterCoefficients m_highpassCoeffs;
    
    /// Cờ đánh dấu hệ số bộ lọc đã được tính
    bool m_filtersInitialized;
};

// ============================================================================
// INLINE UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Tính giá trị tuyệt đối tối đa trong vector
 * @param samples Vector mẫu
 * @return Giá trị tuyệt đối lớn nhất
 */
inline float findMaxAbsValue(const std::vector<float>& samples) {
    float maxVal = 0.0f;
    for (const auto& s : samples) {
        float absVal = std::fabs(s);
        if (absVal > maxVal) {
            maxVal = absVal;
        }
    }
    return maxVal;
}

/**
 * @brief Chuyển đổi tần số sang tần số góc chuẩn hóa
 * @param freq Tần số (Hz)
 * @param sampleRate Tần số lấy mẫu (Hz)
 * @return Tần số góc chuẩn hóa (rad/sample)
 */
inline double freqToNormalizedOmega(double freq, double sampleRate) {
    return 2.0 * M_PI * freq / sampleRate;
}

/**
 * @brief Pre-warping cho biến đổi bilinear
 * @param omega Tần số góc chuẩn hóa
 * @return Tần số analog tương đương
 */
inline double prewarp(double omega) {
    return 2.0 * std::tan(omega / 2.0);
}

/**
 * @brief Chuyển RespiratoryLabel thành string
 */
inline std::string labelToString(RespiratoryLabel label) {
    switch (label) {
        case RespiratoryLabel::NORMAL: return "Normal";
        case RespiratoryLabel::CRACKLE: return "Crackle";
        case RespiratoryLabel::WHEEZE: return "Wheeze";
        case RespiratoryLabel::BOTH: return "Both";
        default: return "Unknown";
    }
}

// ============================================================================
// DATASET MANAGER CLASS
// ============================================================================

/**
 * @class DatasetManager
 * @brief Quản lý và xử lý ICBHI 2017 Dataset
 * 
 * Class này chịu trách nhiệm:
 * 1. Quét thư mục dataset để tìm các file WAV và TXT
 * 2. Parse file annotation (.txt) để lấy thông tin chu kỳ hô hấp
 * 3. Trích xuất thông tin bệnh nhân từ tên file
 * 4. Tích hợp với SignalProcessor để tiền xử lý tín hiệu
 * 5. Cắt tín hiệu thành các chu kỳ theo annotation
 * 6. Trả về vector các LabeledBreathingCycle để training/inference
 * 
 * Cách sử dụng:
 * @code
 *   DatasetManager manager("data/samples/ICBHI_final_database");
 *   std::vector<LabeledBreathingCycle> dataset;
 *   
 *   if (manager.loadDataset(dataset)) {
 *       std::cout << "Loaded " << dataset.size() << " cycles" << std::endl;
 *       manager.getStatistics().print();
 *   }
 * @endcode
 */
class DatasetManager {
public:
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================
    
    /**
     * @brief Constructor với đường dẫn dataset
     * @param datasetPath Đường dẫn đến thư mục chứa file WAV và TXT
     */
    explicit DatasetManager(const std::string& datasetPath = DEFAULT_DATASET_PATH);
    
    /**
     * @brief Destructor
     */
    ~DatasetManager();
    
    // ========================================================================
    // MAIN METHODS - Phương thức chính
    // ========================================================================
    
    /**
     * @brief Load toàn bộ dataset và xử lý
     * 
     * Quy trình:
     * 1. Quét thư mục tìm tất cả file .wav
     * 2. Với mỗi file .wav:
     *    - Đọc file annotation .txt tương ứng
     *    - Load và tiền xử lý tín hiệu (resample, filter, normalize)
     *    - Cắt thành các chu kỳ theo annotation
     *    - Gán nhãn từ annotation
     * 3. Trả về vector tất cả chu kỳ đã gán nhãn
     * 
     * @param outputCycles Vector chứa kết quả (output)
     * @param verbose In thông tin chi tiết trong quá trình xử lý
     * @return true nếu load thành công ít nhất 1 file
     */
    bool loadDataset(std::vector<LabeledBreathingCycle>& outputCycles, 
                     bool verbose = true);
    
    /**
     * @brief Load và xử lý một file đơn lẻ
     * 
     * @param wavFilePath Đường dẫn file WAV
     * @param outputCycles Vector chứa các chu kỳ từ file này
     * @return true nếu xử lý thành công
     */
    bool processFile(const std::string& wavFilePath,
                     std::vector<LabeledBreathingCycle>& outputCycles);
    
    /**
     * @brief Load dataset với progress callback
     * 
     * @param outputCycles Vector chứa kết quả
     * @param progressCallback Callback function(current, total) để báo tiến độ
     * @return true nếu load thành công
     */
    bool loadDatasetWithProgress(
        std::vector<LabeledBreathingCycle>& outputCycles,
        std::function<void(size_t current, size_t total)> progressCallback);
    
    // ========================================================================
    // ANNOTATION METHODS - Đọc file annotation
    // ========================================================================
    
    /**
     * @brief Đọc file annotation ICBHI (.txt)
     * 
     * Format: [Start] [End] [Crackles] [Wheezes]
     * Mỗi dòng là một chu kỳ hô hấp
     * 
     * @param annotationPath Đường dẫn file .txt
     * @param annotations Vector chứa các annotation đọc được
     * @return true nếu đọc thành công
     */
    bool readAnnotationFile(const std::string& annotationPath,
                            std::vector<CycleAnnotation>& annotations);
    
    /**
     * @brief Parse thông tin bệnh nhân từ tên file
     * 
     * Format: {PatientID}_{RecordingIndex}_{ChestLocation}_{Mode}_{Equipment}
     * 
     * @param filename Tên file (không bao gồm đường dẫn)
     * @param info Struct chứa thông tin parse được
     * @return true nếu parse thành công
     */
    bool parseFilename(const std::string& filename, PatientInfo& info);
    
    // ========================================================================
    // SEGMENTATION - Cắt tín hiệu theo annotation
    // ========================================================================
    
    /**
     * @brief Cắt tín hiệu đã xử lý thành các chu kỳ theo annotation
     * 
     * Dựa vào mốc thời gian [startTime, endTime] trong annotation,
     * trích xuất đoạn tín hiệu tương ứng.
     * 
     * @param processedSignal Tín hiệu đã tiền xử lý (4kHz, filtered, normalized)
     * @param sampleRate Tần số lấy mẫu của tín hiệu (4kHz)
     * @param annotations Danh sách annotation từ file .txt
     * @param patientInfo Thông tin bệnh nhân
     * @param sourceFile Tên file nguồn
     * @param outputCycles Vector chứa các chu kỳ đã cắt và gán nhãn
     */
    void segmentByAnnotations(const std::vector<float>& processedSignal,
                              float sampleRate,
                              const std::vector<CycleAnnotation>& annotations,
                              const PatientInfo& patientInfo,
                              const std::string& sourceFile,
                              std::vector<LabeledBreathingCycle>& outputCycles);
    
    // ========================================================================
    // UTILITY METHODS
    // ========================================================================
    
    /**
     * @brief Lấy thống kê dataset đã load
     * @return DatasetStatistics struct
     */
    const DatasetStatistics& getStatistics() const;
    
    /**
     * @brief Lấy danh sách file WAV trong dataset
     * @return Vector chứa đường dẫn các file WAV
     */
    std::vector<std::string> getWavFiles() const;
    
    /**
     * @brief Kiểm tra dataset path có hợp lệ không
     * @return true nếu thư mục tồn tại và có file WAV
     */
    bool isValidDatasetPath() const;
    
    /**
     * @brief Đặt đường dẫn dataset mới
     * @param path Đường dẫn mới
     */
    void setDatasetPath(const std::string& path);
    
    /**
     * @brief Lấy đường dẫn dataset hiện tại
     * @return Đường dẫn dataset
     */
    const std::string& getDatasetPath() const;
    
    /**
     * @brief Lấy file annotation tương ứng với file WAV
     * @param wavPath Đường dẫn file WAV
     * @return Đường dẫn file annotation (.txt)
     */
    static std::string getAnnotationPath(const std::string& wavPath);
    
    /**
     * @brief Lọc dataset theo nhãn
     * @param allCycles Tất cả chu kỳ
     * @param label Nhãn cần lọc
     * @param filteredCycles Vector chứa kết quả lọc
     */
    static void filterByLabel(const std::vector<LabeledBreathingCycle>& allCycles,
                              RespiratoryLabel label,
                              std::vector<LabeledBreathingCycle>& filteredCycles);
    
    /**
     * @brief Chia dataset thành train/test sets
     * @param allCycles Tất cả chu kỳ
     * @param trainRatio Tỷ lệ train (0.0 - 1.0)
     * @param trainSet Vector chứa train set
     * @param testSet Vector chứa test set
     * @param shuffleSeed Seed cho random shuffle (-1 = không shuffle)
     */
    static void splitDataset(const std::vector<LabeledBreathingCycle>& allCycles,
                             float trainRatio,
                             std::vector<LabeledBreathingCycle>& trainSet,
                             std::vector<LabeledBreathingCycle>& testSet,
                             int shuffleSeed = -1);

private:
    // ========================================================================
    // PRIVATE MEMBERS
    // ========================================================================
    
    std::string m_datasetPath;              ///< Đường dẫn thư mục dataset
    DatasetStatistics m_statistics;         ///< Thống kê dataset
    std::unique_ptr<SignalProcessor> m_signalProcessor; ///< Signal processor instance
    
    /**
     * @brief Quét thư mục tìm tất cả file WAV
     * @param wavFiles Vector chứa đường dẫn các file WAV tìm được
     * @return Số file tìm được
     */
    size_t scanDirectory(std::vector<std::string>& wavFiles) const;
    
    /**
     * @brief Cập nhật thống kê từ một chu kỳ
     */
    void updateStatistics(const LabeledBreathingCycle& cycle);
    
    /**
     * @brief Reset thống kê
     */
    void resetStatistics();
};

} // namespace respiratory

#endif // SIGNAL_PREP_HPP
