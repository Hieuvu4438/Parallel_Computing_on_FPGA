/**
 * @file WaveletTransform.hpp
 * @brief Discrete Wavelet Transform (DWT) for Multi-resolution Spectrogram Generation
 * 
 * Triển khai biến đổi Wavelet cho Layer 4 của Cascaded Framework.
 * Sử dụng Continuous Wavelet Transform (CWT) với Morlet wavelet để tạo
 * ảnh Spectrogram đa phân giải làm đầu vào cho CNN.
 * 
 * Ưu điểm so với STFT truyền thống:
 * - Độ phân giải thời gian-tần số thích ứng
 * - Phù hợp với tín hiệu non-stationary (âm thanh hô hấp)
 * - Capture được cả transient (crackles) và continuous (wheeze)
 * 
 * @author Research Team
 * @date 2026
 */

#ifndef WAVELET_TRANSFORM_HPP
#define WAVELET_TRANSFORM_HPP

#include <vector>
#include <array>
#include <complex>
#include <cstdint>
#include <cmath>
#include <memory>
#include <string>

namespace respiratory {

// ============================================================================
// CONSTANTS
// ============================================================================

/// Kích thước ảnh đầu ra cho CNN (vuông)
constexpr int SPECTROGRAM_SIZE = 224;  // Phù hợp với MobileNetV2

/// Số scales mặc định cho CWT
constexpr int DEFAULT_NUM_SCALES = 64;

/// Tần số trung tâm của Morlet wavelet (ω₀)
constexpr float MORLET_CENTER_FREQUENCY = 6.0f;

/// Sample rate mặc định (Hz)
constexpr int WAVELET_SAMPLE_RATE = 4000;

// ============================================================================
// ENUMS
// ============================================================================

/**
 * @enum WaveletType
 * @brief Loại wavelet sử dụng
 */
enum class WaveletType {
    MORLET,         ///< Morlet (Gabor) wavelet - tốt cho phân tích tần số
    MEXICAN_HAT,    ///< Mexican Hat (Ricker) - tốt cho phát hiện transient
    PAUL,           ///< Paul wavelet - cân bằng thời gian-tần số
    DOG             ///< Derivative of Gaussian
};

/**
 * @enum NormalizationType
 * @brief Phương pháp chuẩn hóa spectrogram
 */
enum class NormalizationType {
    MIN_MAX,        ///< Chuẩn hóa về [0, 1]
    Z_SCORE,        ///< Standardization (mean=0, std=1)
    LOG_SCALE,      ///< Log transform + normalization
    POWER_TO_DB     ///< Chuyển power sang dB
};

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @struct WaveletConfig
 * @brief Cấu hình cho Wavelet Transform
 */
struct WaveletConfig {
    WaveletType waveletType = WaveletType::MORLET;
    NormalizationType normType = NormalizationType::LOG_SCALE;
    
    int numScales = DEFAULT_NUM_SCALES;     ///< Số scales (tần số)
    float minFreq = 50.0f;                  ///< Tần số thấp nhất (Hz)
    float maxFreq = 2000.0f;                ///< Tần số cao nhất (Hz)
    int sampleRate = WAVELET_SAMPLE_RATE;
    
    int outputWidth = SPECTROGRAM_SIZE;     ///< Chiều rộng ảnh đầu ra
    int outputHeight = SPECTROGRAM_SIZE;    ///< Chiều cao ảnh đầu ra
    int numChannels = 1;                    ///< Số kênh (1=grayscale, 3=RGB)
    
    float morletOmega0 = MORLET_CENTER_FREQUENCY;  ///< Tần số Morlet
    
    WaveletConfig() = default;
};

/**
 * @struct Spectrogram
 * @brief Kết quả spectrogram
 */
struct Spectrogram {
    std::vector<float> data;        ///< Dữ liệu ảnh (row-major, normalized)
    int width;                      ///< Chiều rộng
    int height;                     ///< Chiều cao
    int channels;                   ///< Số kênh
    
    std::vector<float> frequencies; ///< Tần số tương ứng với mỗi row
    std::vector<float> times;       ///< Thời gian tương ứng với mỗi column
    
    float minValue;                 ///< Giá trị min (trước normalize)
    float maxValue;                 ///< Giá trị max (trước normalize)
    
    Spectrogram() : width(0), height(0), channels(1), 
                    minValue(0.0f), maxValue(0.0f) {}
    
    /**
     * @brief Lấy giá trị tại vị trí (x, y, c)
     */
    float at(int x, int y, int c = 0) const {
        return data[(y * width + x) * channels + c];
    }
    
    /**
     * @brief Set giá trị tại vị trí (x, y, c)
     */
    void set(int x, int y, float value, int c = 0) {
        data[(y * width + x) * channels + c] = value;
    }
    
    /**
     * @brief Kích thước tổng của data
     */
    size_t size() const { return data.size(); }
    
    /**
     * @brief Cấp phát bộ nhớ
     */
    void allocate(int w, int h, int c = 1) {
        width = w;
        height = h;
        channels = c;
        data.resize(w * h * c, 0.0f);
    }
    
    /**
     * @brief Chuyển đổi sang vector flat cho CNN input
     * @return Vector float đã chuẩn hóa [0, 1]
     */
    std::vector<float> toFlatVector() const { return data; }
    
    /**
     * @brief Chuyển đổi sang format NCHW cho ONNX
     * @return Vector với layout [1, C, H, W]
     */
    std::vector<float> toNCHW() const;
    
    /**
     * @brief Chuyển đổi sang format NHWC
     * @return Vector với layout [1, H, W, C]
     */
    std::vector<float> toNHWC() const;
};

/**
 * @struct CWTCoefficients
 * @brief Hệ số Continuous Wavelet Transform
 */
struct CWTCoefficients {
    std::vector<std::vector<std::complex<float>>> coeffs; ///< [scale][time]
    std::vector<float> scales;                            ///< Scales đã sử dụng
    std::vector<float> frequencies;                       ///< Pseudo-frequencies
    int numScales;
    int numTimePoints;
    
    CWTCoefficients() : numScales(0), numTimePoints(0) {}
    
    /**
     * @brief Lấy magnitude tại (scale, time)
     */
    float getMagnitude(int scale, int time) const {
        return std::abs(coeffs[scale][time]);
    }
    
    /**
     * @brief Lấy power tại (scale, time)
     */
    float getPower(int scale, int time) const {
        float mag = getMagnitude(scale, time);
        return mag * mag;
    }
};

// ============================================================================
// WAVELET TRANSFORM CLASS
// ============================================================================

/**
 * @class WaveletTransform
 * @brief Triển khai Continuous Wavelet Transform cho phân tích âm thanh hô hấp
 * 
 * Pipeline:
 * 1. Tính toán scales từ min/max frequency
 * 2. Áp dụng CWT với Morlet wavelet
 * 3. Tính scalogram (magnitude/power)
 * 4. Resize về kích thước đầu ra mong muốn
 * 5. Normalize cho CNN input
 * 
 * Thiết kế FPGA-friendly:
 * - Sử dụng FFT-based convolution cho hiệu suất
 * - Có thể port sang HLS với fixed-point
 * - Memory access patterns tối ưu cho BRAM
 */
class WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param config Cấu hình transform
     */
    explicit WaveletTransform(const WaveletConfig& config = WaveletConfig());
    
    /**
     * @brief Destructor
     */
    ~WaveletTransform();
    
    // ========================================================================
    // MAIN TRANSFORM INTERFACE
    // ========================================================================
    
    /**
     * @brief Biến đổi tín hiệu thành spectrogram cho CNN
     * 
     * @param signal Tín hiệu đầu vào (đã normalize)
     * @param spectrogram Output spectrogram
     * @return true nếu thành công
     */
    bool transform(const std::vector<float>& signal, Spectrogram& spectrogram);
    
    /**
     * @brief Tính CWT coefficients (intermediate result)
     * 
     * @param signal Tín hiệu đầu vào
     * @param coeffs Output coefficients
     * @return true nếu thành công
     */
    bool computeCWT(const std::vector<float>& signal, CWTCoefficients& coeffs);
    
    /**
     * @brief Chuyển CWT coefficients thành spectrogram
     * 
     * @param coeffs CWT coefficients
     * @param spectrogram Output spectrogram
     * @return true nếu thành công
     */
    bool coefficientsToSpectrogram(const CWTCoefficients& coeffs, 
                                    Spectrogram& spectrogram);
    
    // ========================================================================
    // BATCH PROCESSING
    // ========================================================================
    
    /**
     * @brief Transform batch nhiều signals (parallel)
     */
    std::vector<Spectrogram> transformBatch(
        const std::vector<std::vector<float>>& signals);
    
    // ========================================================================
    // CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Đặt cấu hình mới
     */
    void setConfig(const WaveletConfig& config);
    
    /**
     * @brief Lấy cấu hình hiện tại
     */
    const WaveletConfig& getConfig() const { return m_config; }
    
    /**
     * @brief Đặt loại wavelet
     */
    void setWaveletType(WaveletType type);
    
    /**
     * @brief Đặt phương pháp normalize
     */
    void setNormalizationType(NormalizationType type);
    
    /**
     * @brief Đặt kích thước output
     */
    void setOutputSize(int width, int height);
    
    /**
     * @brief Đặt dải tần số
     */
    void setFrequencyRange(float minFreq, float maxFreq);
    
    // ========================================================================
    // UTILITY
    // ========================================================================
    
    /**
     * @brief Tính pseudo-frequency từ scale
     */
    float scaleToFrequency(float scale) const;
    
    /**
     * @brief Tính scale từ frequency
     */
    float frequencyToScale(float frequency) const;
    
    /**
     * @brief Lấy danh sách scales đã tính
     */
    const std::vector<float>& getScales() const { return m_scales; }
    
    /**
     * @brief Lấy danh sách frequencies tương ứng
     */
    const std::vector<float>& getFrequencies() const { return m_frequencies; }

private:
    WaveletConfig m_config;
    
    // Pre-computed values
    std::vector<float> m_scales;
    std::vector<float> m_frequencies;
    bool m_isInitialized;
    
    // ========================================================================
    // PRIVATE METHODS
    // ========================================================================
    
    /**
     * @brief Khởi tạo scales và frequencies
     */
    void initializeScales();
    
    /**
     * @brief Tính Morlet wavelet function
     */
    std::vector<std::complex<float>> computeMorletWavelet(float scale, int length);
    
    /**
     * @brief Tính Mexican Hat wavelet function
     */
    std::vector<float> computeMexicanHat(float scale, int length);
    
    /**
     * @brief Convolution với wavelet
     */
    std::vector<std::complex<float>> convolveWithWavelet(
        const std::vector<float>& signal,
        const std::vector<std::complex<float>>& wavelet);
    
    /**
     * @brief FFT-based convolution (faster for long signals)
     */
    std::vector<std::complex<float>> fftConvolve(
        const std::vector<float>& signal,
        const std::vector<std::complex<float>>& wavelet);
    
    /**
     * @brief Resize scalogram về kích thước mong muốn
     */
    void resizeScalogram(const std::vector<std::vector<float>>& input,
                         std::vector<std::vector<float>>& output,
                         int newWidth, int newHeight);
    
    /**
     * @brief Bilinear interpolation cho resize
     */
    float bilinearInterpolate(const std::vector<std::vector<float>>& data,
                              float x, float y);
    
    /**
     * @brief Normalize spectrogram
     */
    void normalizeSpectrogram(Spectrogram& spectrogram);
    
    /**
     * @brief Log transform
     */
    void applyLogTransform(Spectrogram& spectrogram);
    
    /**
     * @brief Power to dB conversion
     */
    void powerToDb(Spectrogram& spectrogram, float refPower = 1.0f);
    
    // ========================================================================
    // FFT HELPERS (Simple DFT for now, can be replaced with FFTW)
    // ========================================================================
    
    /**
     * @brief Simple DFT implementation
     */
    std::vector<std::complex<float>> dft(const std::vector<std::complex<float>>& input);
    
    /**
     * @brief Simple IDFT implementation
     */
    std::vector<std::complex<float>> idft(const std::vector<std::complex<float>>& input);
    
    /**
     * @brief Next power of 2
     */
    int nextPowerOf2(int n);
    
    /**
     * @brief Zero-pad signal to length
     */
    std::vector<std::complex<float>> zeroPad(
        const std::vector<std::complex<float>>& input, int newLength);
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Tạo spectrogram từ tín hiệu với cấu hình mặc định
 */
Spectrogram createSpectrogram(const std::vector<float>& signal,
                               int sampleRate = WAVELET_SAMPLE_RATE);

/**
 * @brief So sánh hai spectrograms (similarity score)
 */
float compareSpectrograms(const Spectrogram& a, const Spectrogram& b);

/**
 * @brief Lưu spectrogram ra file (binary format)
 */
bool saveSpectrogram(const Spectrogram& spec, const std::string& filename);

/**
 * @brief Load spectrogram từ file
 */
bool loadSpectrogram(Spectrogram& spec, const std::string& filename);

/**
 * @brief Export spectrogram ra CSV (for debugging)
 */
bool exportSpectrogramToCSV(const Spectrogram& spec, const std::string& filename);

} // namespace respiratory

#endif // WAVELET_TRANSFORM_HPP
