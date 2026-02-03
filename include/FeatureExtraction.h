/**
 * @file FeatureExtraction.h
 * @brief Feature Extraction Module for Respiratory Sound Analysis
 * 
 * Implements feature extraction based on IEEE paper:
 * "Cascaded Framework with Hardware Acceleration for Respiratory Sound Analysis
 *  on Heterogeneous FPGA"
 * 
 * Features extracted:
 * 1. Time-Domain Features:
 *    - EED (Extreme Energy Difference)
 *    - ZCR (Zero Crossing Rate)
 *    - RMSE (Root Mean Square Energy)
 * 
 * 2. Frequency/Cepstral Features:
 *    - MFCC (39-dimensional: 13 static + 13 delta + 13 delta-delta)
 * 
 * @author Research Team
 * @date 2026
 */

#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>
#include <string>
#include <complex>

namespace respiratory {

// ============================================================================
// CONSTANTS - Thông số trích xuất đặc trưng theo bài báo IEEE
// ============================================================================

/// Tần số lấy mẫu (Hz) - phải khớp với SignalPrep
constexpr float FEATURE_SAMPLE_RATE = 4000.0f;

/// Kích thước frame (ms) - 25ms theo bài báo
constexpr float FRAME_SIZE_MS = 25.0f;

/// Độ chồng lấp (overlap) - 50%
constexpr float FRAME_OVERLAP_RATIO = 0.5f;

/// Kích thước frame tính bằng samples (25ms @ 4kHz = 100 samples)
constexpr int FRAME_SIZE_SAMPLES = static_cast<int>(FRAME_SIZE_MS * FEATURE_SAMPLE_RATE / 1000.0f);

/// Hop size (samples) - 50% overlap = 50 samples
constexpr int HOP_SIZE_SAMPLES = static_cast<int>(FRAME_SIZE_SAMPLES * (1.0f - FRAME_OVERLAP_RATIO));

/// Kích thước FFT (power of 2 >= frame size)
constexpr int FFT_SIZE = 256;

/// Số lượng Mel filters
constexpr int NUM_MEL_FILTERS = 26;

/// Số lượng MFCC coefficients (không tính delta)
constexpr int NUM_MFCC_COEFFS = 13;

/// Tổng số MFCC features (static + delta + delta-delta)
constexpr int TOTAL_MFCC_FEATURES = NUM_MFCC_COEFFS * 3;  // 39

/// Pre-emphasis coefficient
constexpr float PRE_EMPHASIS_COEFF = 0.97f;

/// Tần số thấp nhất cho Mel filterbank (Hz)
constexpr float MEL_LOW_FREQ = 50.0f;

/// Tần số cao nhất cho Mel filterbank (Hz) - Nyquist/2
constexpr float MEL_HIGH_FREQ = 2000.0f;

/// Delta window size cho tính delta coefficients
constexpr int DELTA_WINDOW = 2;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @struct FrameFeatures
 * @brief Đặc trưng của một frame đơn lẻ
 */
struct FrameFeatures {
    std::vector<float> mfcc;        ///< MFCC coefficients (13 chiều)
    float zcr;                       ///< Zero Crossing Rate
    float rmse;                      ///< Root Mean Square Energy
    float energy;                    ///< Frame energy
    
    FrameFeatures() : zcr(0.0f), rmse(0.0f), energy(0.0f) {
        mfcc.resize(NUM_MFCC_COEFFS, 0.0f);
    }
};

/**
 * @struct CycleFeatures
 * @brief Tất cả đặc trưng của một chu kỳ hô hấp hoàn chỉnh
 */
struct CycleFeatures {
    // Time-domain features (global cho toàn cycle)
    float eed;                       ///< Extreme Energy Difference
    float zcr_mean;                  ///< Mean Zero Crossing Rate
    float zcr_std;                   ///< Std Zero Crossing Rate
    float rmse_mean;                 ///< Mean RMSE
    float rmse_std;                  ///< Std RMSE
    
    // MFCC features (averaged over all frames)
    std::vector<float> mfcc_mean;    ///< Mean MFCC (13 chiều)
    std::vector<float> mfcc_std;     ///< Std MFCC (13 chiều)
    std::vector<float> delta_mean;   ///< Mean Delta MFCC (13 chiều)
    std::vector<float> delta_std;    ///< Std Delta MFCC (13 chiều)
    std::vector<float> delta2_mean;  ///< Mean Delta-Delta MFCC (13 chiều)
    std::vector<float> delta2_std;   ///< Std Delta-Delta MFCC (13 chiều)
    
    // Frame-level features (optional, for detailed analysis)
    std::vector<FrameFeatures> frameFeatures;
    
    // Metadata
    int numFrames;                   ///< Số frame đã xử lý
    float durationSec;               ///< Thời lượng (giây)
    
    CycleFeatures() : eed(0.0f), zcr_mean(0.0f), zcr_std(0.0f),
                      rmse_mean(0.0f), rmse_std(0.0f), numFrames(0), durationSec(0.0f) {
        mfcc_mean.resize(NUM_MFCC_COEFFS, 0.0f);
        mfcc_std.resize(NUM_MFCC_COEFFS, 0.0f);
        delta_mean.resize(NUM_MFCC_COEFFS, 0.0f);
        delta_std.resize(NUM_MFCC_COEFFS, 0.0f);
        delta2_mean.resize(NUM_MFCC_COEFFS, 0.0f);
        delta2_std.resize(NUM_MFCC_COEFFS, 0.0f);
    }
    
    /**
     * @brief Chuyển đổi thành vector phẳng để nạp vào classifier
     * 
     * Format output:
     * [EED, ZCR_mean, ZCR_std, RMSE_mean, RMSE_std,
     *  MFCC_mean(13), MFCC_std(13),
     *  Delta_mean(13), Delta_std(13),
     *  Delta2_mean(13), Delta2_std(13)]
     * 
     * Total: 5 + 13*6 = 83 features
     * 
     * @return Vector phẳng chứa tất cả features
     */
    std::vector<float> toFlatVector() const;
    
    /**
     * @brief Lấy số chiều của feature vector
     */
    static int getFeatureDimension();
    
    /**
     * @brief Lấy tên các features
     */
    static std::vector<std::string> getFeatureNames();
};

// ============================================================================
// MAIN CLASS
// ============================================================================

/**
 * @class FeatureExtractor
 * @brief Trích xuất đặc trưng từ tín hiệu âm thanh hô hấp
 * 
 * Lớp này thực hiện:
 * 1. Chia tín hiệu thành frames (25ms, 50% overlap)
 * 2. Tính time-domain features (EED, ZCR, RMSE)
 * 3. Tính MFCC và delta coefficients
 * 4. Aggregate features từ tất cả frames
 * 
 * Thiết kế tương thích với Vitis HLS:
 * - Sử dụng lookup tables cho log và sin/cos
 * - Tránh cấp phát động trong vòng lặp chính
 * - Hỗ trợ fixed-point conversion sau này
 */
class FeatureExtractor {
public:
    // ========================================================================
    // CONSTRUCTOR & DESTRUCTOR
    // ========================================================================
    
    /**
     * @brief Constructor - khởi tạo và precompute các bảng lookup
     * @param sampleRate Tần số lấy mẫu (mặc định 4kHz)
     */
    explicit FeatureExtractor(float sampleRate = FEATURE_SAMPLE_RATE);
    
    /**
     * @brief Destructor
     */
    ~FeatureExtractor();
    
    // ========================================================================
    // MAIN EXTRACTION METHODS
    // ========================================================================
    
    /**
     * @brief Trích xuất tất cả đặc trưng từ một chu kỳ hô hấp
     * 
     * Đây là hàm chính để sử dụng. Input là tín hiệu đã được
     * tiền xử lý (resampled, filtered, normalized).
     * 
     * @param samples Vector mẫu tín hiệu (normalized [-1, 1])
     * @param features Struct chứa tất cả đặc trưng đầu ra
     * @param keepFrameFeatures Giữ lại features từng frame (tốn memory)
     * @return true nếu trích xuất thành công
     */
    bool extractFeatures(const std::vector<float>& samples,
                         CycleFeatures& features,
                         bool keepFrameFeatures = false);
    
    /**
     * @brief Trích xuất features và trả về dạng flat vector
     * 
     * Tiện lợi để nạp trực tiếp vào classifier.
     * 
     * @param samples Vector mẫu tín hiệu
     * @param flatFeatures Vector đầu ra (được resize tự động)
     * @return true nếu thành công
     */
    bool extractFlatFeatures(const std::vector<float>& samples,
                             std::vector<float>& flatFeatures);
    
    // ========================================================================
    // TIME-DOMAIN FEATURE EXTRACTION
    // ========================================================================
    
    /**
     * @brief Tính EED (Extreme Energy Difference)
     * 
     * Công thức: EED = |(w_max^T * x)^2 - (w_min^T * x)^2|
     * 
     * Trong đó w_max và w_min là các weight vectors được tính
     * từ eigenvectors của ma trận hiệp phương sai.
     * 
     * Simplified version: Sử dụng max và min energy trong các frames
     * EED = |E_max - E_min|
     * 
     * @param samples Vector mẫu
     * @return Giá trị EED
     */
    float computeEED(const std::vector<float>& samples);
    
    /**
     * @brief Tính ZCR (Zero Crossing Rate)
     * 
     * Công thức: ZCR = (1/2(N-1)) * sum(|sgn[x(n)] - sgn[x(n-1)]|)
     * 
     * @param samples Vector mẫu
     * @return Giá trị ZCR (trong khoảng [0, 1])
     */
    float computeZCR(const std::vector<float>& samples);
    
    /**
     * @brief Tính RMSE (Root Mean Square Energy)
     * 
     * Công thức: RMSE = sqrt((1/N) * sum(|x(n)|^2))
     * 
     * @param samples Vector mẫu
     * @return Giá trị RMSE
     */
    float computeRMSE(const std::vector<float>& samples);
    
    // ========================================================================
    // MFCC EXTRACTION
    // ========================================================================
    
    /**
     * @brief Trích xuất MFCC từ một frame
     * 
     * Quy trình:
     * 1. Pre-emphasis: y[n] = x[n] - α*x[n-1]
     * 2. Hamming window: w[n] = 0.54 - 0.46*cos(2πn/(N-1))
     * 3. FFT
     * 4. Power spectrum: |X[k]|^2
     * 5. Mel filterbank
     * 6. Log compression
     * 7. DCT
     * 
     * @param frame Vector mẫu của frame (kích thước FRAME_SIZE_SAMPLES)
     * @param mfcc Vector output chứa MFCC coefficients (13 chiều)
     */
    void extractFrameMFCC(const std::vector<float>& frame,
                          std::vector<float>& mfcc);
    
    /**
     * @brief Tính Delta coefficients
     * 
     * Công thức: d[t] = (sum_{n=1}^{N} n*(c[t+n] - c[t-n])) / (2*sum_{n=1}^{N} n^2)
     * 
     * @param coeffs Ma trận MFCC (frames x coefficients)
     * @param delta Ma trận Delta output
     * @param windowSize Kích thước cửa sổ (mặc định 2)
     */
    void computeDeltaCoeffs(const std::vector<std::vector<float>>& coeffs,
                            std::vector<std::vector<float>>& delta,
                            int windowSize = DELTA_WINDOW);
    
    // ========================================================================
    // DSP HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Áp dụng pre-emphasis filter
     * 
     * y[n] = x[n] - α*x[n-1], với α = 0.97
     * 
     * @param samples Input samples
     * @param output Output samples
     * @param coeff Pre-emphasis coefficient
     */
    void applyPreEmphasis(const std::vector<float>& samples,
                          std::vector<float>& output,
                          float coeff = PRE_EMPHASIS_COEFF);
    
    /**
     * @brief Áp dụng Hamming window
     * 
     * w[n] = 0.54 - 0.46*cos(2πn/(N-1))
     * 
     * @param samples Input/Output samples (in-place)
     */
    void applyHammingWindow(std::vector<float>& samples);
    
    /**
     * @brief Tính FFT (sử dụng FFTW3 nếu có, hoặc fallback DFT)
     * 
     * @param samples Input samples (time domain)
     * @param spectrum Output spectrum (complex)
     */
    void computeFFT(const std::vector<float>& samples,
                    std::vector<std::complex<float>>& spectrum);
    
    /**
     * @brief Tính power spectrum từ FFT output
     * 
     * P[k] = |X[k]|^2 / N
     * 
     * @param spectrum Complex FFT output
     * @param powerSpectrum Output power spectrum
     */
    void computePowerSpectrum(const std::vector<std::complex<float>>& spectrum,
                              std::vector<float>& powerSpectrum);
    
    /**
     * @brief Áp dụng Mel filterbank
     * 
     * @param powerSpectrum Input power spectrum
     * @param melEnergies Output Mel energies (NUM_MEL_FILTERS chiều)
     */
    void applyMelFilterbank(const std::vector<float>& powerSpectrum,
                            std::vector<float>& melEnergies);
    
    /**
     * @brief Tính DCT (Discrete Cosine Transform)
     * 
     * c[k] = sum_{n=0}^{N-1} x[n] * cos(π*k*(2n+1)/(2N))
     * 
     * @param input Input vector
     * @param output Output DCT coefficients
     * @param numCoeffs Số coefficients cần lấy
     */
    void computeDCT(const std::vector<float>& input,
                    std::vector<float>& output,
                    int numCoeffs = NUM_MFCC_COEFFS);
    
    // ========================================================================
    // UTILITY METHODS
    // ========================================================================
    
    /**
     * @brief Chuyển đổi Hz sang Mel scale
     * 
     * m = 2595 * log10(1 + f/700)
     * 
     * @param freq Tần số Hz
     * @return Tần số Mel
     */
    static float hzToMel(float freq);
    
    /**
     * @brief Chuyển đổi Mel sang Hz
     * 
     * f = 700 * (10^(m/2595) - 1)
     * 
     * @param mel Tần số Mel
     * @return Tần số Hz
     */
    static float melToHz(float mel);
    
    /**
     * @brief Sign function
     * @return 1 nếu x > 0, -1 nếu x < 0, 0 nếu x == 0
     */
    static int sign(float x);
    
    /**
     * @brief Lấy sample rate hiện tại
     */
    float getSampleRate() const { return m_sampleRate; }
    
    /**
     * @brief Lấy frame size (samples)
     */
    int getFrameSize() const { return m_frameSize; }
    
    /**
     * @brief Lấy hop size (samples)
     */
    int getHopSize() const { return m_hopSize; }

private:
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Khởi tạo Hamming window lookup table
     */
    void initHammingWindow();
    
    /**
     * @brief Khởi tạo Mel filterbank
     */
    void initMelFilterbank();
    
    /**
     * @brief Khởi tạo DCT matrix
     */
    void initDCTMatrix();
    
    /**
     * @brief Khởi tạo FFT (nếu dùng FFTW3)
     */
    void initFFT();
    
    // ========================================================================
    // MEMBER VARIABLES
    // ========================================================================
    
    float m_sampleRate;                              ///< Tần số lấy mẫu
    int m_frameSize;                                 ///< Kích thước frame (samples)
    int m_hopSize;                                   ///< Hop size (samples)
    int m_fftSize;                                   ///< FFT size
    
    std::vector<float> m_hammingWindow;              ///< Hamming window coefficients
    std::vector<std::vector<float>> m_melFilterbank; ///< Mel filterbank matrix
    std::vector<std::vector<float>> m_dctMatrix;     ///< DCT matrix
    
    std::vector<int> m_melFilterStart;               ///< Start index của mỗi Mel filter
    std::vector<int> m_melFilterEnd;                 ///< End index của mỗi Mel filter
    
    // FFT buffers
    std::vector<float> m_fftInput;                   ///< FFT input buffer
    std::vector<std::complex<float>> m_fftOutput;    ///< FFT output buffer
    
    bool m_initialized;                              ///< Flag đã khởi tạo
};

// ============================================================================
// BATCH PROCESSING UTILITIES
// ============================================================================

/**
 * @brief Trích xuất features cho batch nhiều cycles
 * 
 * @param extractor FeatureExtractor instance
 * @param allSamples Vector các cycles (mỗi cycle là vector<float>)
 * @param allFeatures Output features cho mỗi cycle
 * @param verbose In progress
 */
void extractBatchFeatures(FeatureExtractor& extractor,
                          const std::vector<std::vector<float>>& allSamples,
                          std::vector<CycleFeatures>& allFeatures,
                          bool verbose = false);

/**
 * @brief Trích xuất flat features cho batch
 * 
 * @param extractor FeatureExtractor instance
 * @param allSamples Vector các cycles
 * @param featureMatrix Output: mỗi hàng là feature vector của 1 cycle
 */
void extractBatchFlatFeatures(FeatureExtractor& extractor,
                              const std::vector<std::vector<float>>& allSamples,
                              std::vector<std::vector<float>>& featureMatrix);

} // namespace respiratory

#endif // FEATURE_EXTRACTION_H
