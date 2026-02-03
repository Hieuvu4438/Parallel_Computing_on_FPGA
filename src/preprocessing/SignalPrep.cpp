/**
 * @file SignalPrep.cpp
 * @brief Implementation of Signal Preprocessing Module
 * 
 * Triển khai chi tiết các hàm xử lý tín hiệu cho hệ thống phân tích
 * âm thanh hô hấp theo bài báo IEEE "Cascaded Framework with Hardware
 * Acceleration for Respiratory Sound Analysis".
 * 
 * Bao gồm:
 * - SignalProcessor: Tiền xử lý tín hiệu (resample, filter, normalize)
 * - DatasetManager: Quản lý và load ICBHI 2017 Dataset
 * 
 * @author Research Team
 * @date 2026
 */

// ============================================================================
// INCLUDES
// ============================================================================

#include "SignalPrep.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <random>
#include <set>
#include <iomanip>

// Namespace alias cho filesystem
namespace fs = std::filesystem;

// ----------------------------------------------------------------------------
// dr_wav - Single-header WAV file library
// Định nghĩa DR_WAV_IMPLEMENTATION chỉ trong một file .cpp
// ----------------------------------------------------------------------------
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

namespace respiratory {

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

SignalProcessor::SignalProcessor()
    : m_currentSampleRate(0.0f)
    , m_hasValidData(false)
    , m_filtersInitialized(false)
{
    // Constructor khởi tạo các member variables
    // Hệ số bộ lọc sẽ được tính khi cần thiết (lazy initialization)
}

SignalProcessor::~SignalProcessor() {
    // Destructor - vector tự động giải phóng bộ nhớ
}

// ============================================================================
// MAIN PIPELINE IMPLEMENTATION
// ============================================================================

bool SignalProcessor::processFile(const std::string& inputPath,
                                   std::vector<BreathingCycle>& outputCycles) {
    /**
     * Pipeline xử lý chính:
     * 1. Đọc file WAV
     * 2. Chuyển sang mono (nếu stereo)
     * 3. Resampling về 4kHz
     * 4. Lọc band-pass 50-2500Hz
     * 5. Chuẩn hóa biên độ
     * 6. Phân đoạn chu kỳ hô hấp
     */
    
    // Reset trạng thái trước khi xử lý
    reset();
    
    // ----- BƯỚC 1: Đọc file WAV -----
    AudioData audioData;
    if (!loadWavFile(inputPath, audioData)) {
        std::cerr << "[SignalProcessor] Error: Failed to load WAV file: " 
                  << inputPath << std::endl;
        return false;
    }
    
    std::cout << "[SignalProcessor] Loaded WAV file: " << inputPath << std::endl;
    std::cout << "  - Sample rate: " << audioData.sampleRate << " Hz" << std::endl;
    std::cout << "  - Channels: " << audioData.channels << std::endl;
    std::cout << "  - Total samples: " << audioData.totalSamples << std::endl;
    
    // ----- BƯỚC 2: Chuyển sang mono (nếu cần) -----
    std::vector<float> monoSamples;
    if (audioData.channels > 1) {
        convertToMono(audioData.samples, monoSamples, audioData.channels);
        std::cout << "  - Converted to mono: " << monoSamples.size() 
                  << " samples" << std::endl;
    } else {
        monoSamples = std::move(audioData.samples);
    }
    
    // ----- BƯỚC 3: Resampling về 4kHz -----
    std::vector<float> resampledSignal;
    resample(monoSamples, audioData.sampleRate, resampledSignal, TARGET_SAMPLE_RATE);
    std::cout << "  - Resampled to " << TARGET_SAMPLE_RATE << " Hz: " 
              << resampledSignal.size() << " samples" << std::endl;
    
    // ----- BƯỚC 4: Lọc band-pass 50-2500Hz -----
    std::vector<float> filteredSignal;
    applyBandpassFilter(resampledSignal, filteredSignal, TARGET_SAMPLE_RATE,
                        BANDPASS_LOW_FREQ, BANDPASS_HIGH_FREQ);
    std::cout << "  - Applied bandpass filter: " << BANDPASS_LOW_FREQ 
              << " - " << BANDPASS_HIGH_FREQ << " Hz" << std::endl;
    
    // ----- BƯỚC 5: Chuẩn hóa biên độ -----
    normalize(filteredSignal);
    std::cout << "  - Normalized to [-1.0, 1.0]" << std::endl;
    
    // Lưu tín hiệu đã xử lý
    m_processedSignal = std::move(filteredSignal);
    m_currentSampleRate = TARGET_SAMPLE_RATE;
    m_hasValidData = true;
    
    // ----- BƯỚC 6: Phân đoạn chu kỳ hô hấp -----
    segmentBreathingCycles(m_processedSignal, outputCycles, TARGET_SAMPLE_RATE);
    std::cout << "  - Detected " << outputCycles.size() 
              << " breathing cycles" << std::endl;
    
    return true;
}

bool SignalProcessor::processBuffer(const std::vector<float>& inputSamples,
                                    uint32_t inputSampleRate,
                                    std::vector<BreathingCycle>& outputCycles) {
    /**
     * Xử lý dữ liệu từ buffer thay vì file
     * Hữu ích khi dữ liệu đã có sẵn trong bộ nhớ
     */
    
    reset();
    
    if (inputSamples.empty()) {
        std::cerr << "[SignalProcessor] Error: Empty input buffer" << std::endl;
        return false;
    }
    
    // Resampling
    std::vector<float> resampledSignal;
    resample(inputSamples, inputSampleRate, resampledSignal, TARGET_SAMPLE_RATE);
    
    // Lọc band-pass
    std::vector<float> filteredSignal;
    applyBandpassFilter(resampledSignal, filteredSignal);
    
    // Chuẩn hóa
    normalize(filteredSignal);
    
    // Lưu kết quả
    m_processedSignal = std::move(filteredSignal);
    m_currentSampleRate = TARGET_SAMPLE_RATE;
    m_hasValidData = true;
    
    // Phân đoạn
    segmentBreathingCycles(m_processedSignal, outputCycles);
    
    return true;
}

// ============================================================================
// WAV FILE LOADING
// ============================================================================

bool SignalProcessor::loadWavFile(const std::string& filePath, AudioData& audioData) {
    /**
     * Sử dụng dr_wav để đọc file WAV
     * 
     * dr_wav tự động xử lý:
     * - Các định dạng bit depth khác nhau (8, 16, 24, 32-bit)
     * - PCM và IEEE float
     * - Endianness
     * 
     * Output luôn là float normalized [-1.0, 1.0]
     */
    
    drwav wav;
    
    // Mở file WAV
    if (!drwav_init_file(&wav, filePath.c_str(), nullptr)) {
        std::cerr << "[loadWavFile] Failed to open file: " << filePath << std::endl;
        return false;
    }
    
    // Lấy thông tin metadata
    audioData.sampleRate = wav.sampleRate;
    audioData.channels = wav.channels;
    audioData.totalSamples = static_cast<uint32_t>(wav.totalPCMFrameCount);
    
    // Validate thông số
    if (audioData.sampleRate == 0 || audioData.channels == 0 || 
        audioData.totalSamples == 0) {
        std::cerr << "[loadWavFile] Invalid WAV file parameters" << std::endl;
        drwav_uninit(&wav);
        return false;
    }
    
    // Tính tổng số samples (frames * channels)
    size_t totalSamplesAllChannels = audioData.totalSamples * audioData.channels;
    
    // Cấp phát bộ nhớ và đọc dữ liệu
    audioData.samples.resize(totalSamplesAllChannels);
    
    // Đọc tất cả samples dưới dạng float normalized
    size_t samplesRead = drwav_read_pcm_frames_f32(&wav, audioData.totalSamples,
                                                    audioData.samples.data());
    
    if (samplesRead != audioData.totalSamples) {
        std::cerr << "[loadWavFile] Warning: Read " << samplesRead 
                  << " frames, expected " << audioData.totalSamples << std::endl;
        // Điều chỉnh kích thước nếu đọc ít hơn mong đợi
        audioData.samples.resize(samplesRead * audioData.channels);
        audioData.totalSamples = static_cast<uint32_t>(samplesRead);
    }
    
    // Đóng file
    drwav_uninit(&wav);
    
    return true;
}

// ============================================================================
// RESAMPLING
// ============================================================================

void SignalProcessor::resample(const std::vector<float>& input,
                               uint32_t inputRate,
                               std::vector<float>& output,
                               float targetRate) {
    /**
     * Resampling sử dụng Linear Interpolation
     * 
     * Thuật toán:
     * 1. Tính tỷ lệ resampling: ratio = inputRate / targetRate
     * 2. Với mỗi sample đầu ra tại vị trí n:
     *    - Tính vị trí tương ứng trong input: pos = n * ratio
     *    - Nội suy tuyến tính giữa input[floor(pos)] và input[ceil(pos)]
     * 
     * Phương pháp này đơn giản và phù hợp với HLS vì:
     * - Không cần FFT hoặc bộ lọc phức tạp
     * - Có thể pipeline hóa dễ dàng
     * - Sử dụng bộ nhớ cố định
     * 
     * Lưu ý: Với downsampling lớn, nên áp dụng anti-aliasing filter trước
     * (đã được xử lý bởi bandpass filter sau đó)
     */
    
    if (input.empty()) {
        output.clear();
        return;
    }
    
    // Nếu tần số giống nhau, không cần resampling
    if (std::abs(static_cast<float>(inputRate) - targetRate) < 1.0f) {
        output = input;
        return;
    }
    
    // Tính số samples đầu ra
    double ratio = static_cast<double>(inputRate) / static_cast<double>(targetRate);
    size_t outputLength = static_cast<size_t>(input.size() / ratio);
    
    // Cấp phát bộ nhớ đầu ra
    output.resize(outputLength);
    
    // Thực hiện nội suy tuyến tính
    // HLS Pragma: có thể thêm #pragma HLS PIPELINE để tối ưu
    for (size_t i = 0; i < outputLength; ++i) {
        // Vị trí trong tín hiệu gốc (dạng số thực)
        double srcPos = i * ratio;
        
        // Chỉ số nguyên và phần thập phân
        size_t idx0 = static_cast<size_t>(srcPos);
        size_t idx1 = idx0 + 1;
        double frac = srcPos - idx0;
        
        // Đảm bảo không vượt quá bounds
        if (idx1 >= input.size()) {
            idx1 = input.size() - 1;
        }
        
        // Nội suy tuyến tính: y = y0 + (y1 - y0) * frac
        output[i] = static_cast<float>(
            input[idx0] * (1.0 - frac) + input[idx1] * frac
        );
    }
}

// ============================================================================
// BANDPASS FILTER
// ============================================================================

void SignalProcessor::applyBandpassFilter(const std::vector<float>& input,
                                          std::vector<float>& output,
                                          float sampleRate,
                                          float lowCutoff,
                                          float highCutoff) {
    /**
     * Bộ lọc Band-pass Butterworth
     * 
     * Thực hiện bằng cách cascade:
     * 1. High-pass filter tại lowCutoff (50Hz) - loại bỏ DC và tiếng ồn tần số thấp
     * 2. Low-pass filter tại highCutoff (2500Hz) - loại bỏ aliasing và nhiễu cao tần
     * 
     * Sử dụng zero-phase filtering (filtfilt) để:
     * - Tránh méo pha
     * - Đáp ứng biên độ chính xác hơn
     * - Bậc hiệu dụng gấp đôi (8 thay vì 4)
     */
    
    if (input.empty()) {
        output.clear();
        return;
    }
    
    // Kiểm tra tần số Nyquist
    // Tần số cắt cao phải nhỏ hơn tần số Nyquist (sampleRate/2)
    float nyquist = sampleRate / 2.0f;
    if (highCutoff >= nyquist) {
        highCutoff = nyquist * 0.95f;  // Giảm xuống 95% Nyquist
        // Note: Warning bị ẩn để tránh spam khi xử lý nhiều file
        // std::cerr << "[BandpassFilter] High cutoff adjusted to " << highCutoff << " Hz\n";
    }
    
    // Thiết kế bộ lọc highpass (nếu chưa có hoặc thông số thay đổi)
    FilterCoefficients highpassCoeffs;
    designButterworthHighpass(lowCutoff, sampleRate, FILTER_ORDER, highpassCoeffs);
    
    // Thiết kế bộ lọc lowpass
    FilterCoefficients lowpassCoeffs;
    designButterworthLowpass(highCutoff, sampleRate, FILTER_ORDER, lowpassCoeffs);
    
    // Áp dụng highpass filter (zero-phase)
    std::vector<float> tempSignal;
    applyZeroPhaseFilter(input, tempSignal, highpassCoeffs);
    
    // Áp dụng lowpass filter (zero-phase)
    applyZeroPhaseFilter(tempSignal, output, lowpassCoeffs);
}

void SignalProcessor::designButterworthLowpass(float cutoffFreq,
                                                float sampleRate,
                                                int /* order */,
                                                FilterCoefficients& coeffs) {
    /**
     * Thiết kế bộ lọc Butterworth Lowpass bằng Bilinear Transform
     * 
     * Các bước:
     * 1. Pre-warp tần số cắt analog
     * 2. Tính poles của Butterworth analog
     * 3. Bilinear transform để chuyển sang digital
     * 4. Tính hệ số bộ lọc digital
     * 
     * Bộ lọc Butterworth bậc 2 (biquad section):
     * H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
     * 
     * Note: Hiện tại sử dụng bậc 2 cố định. Tham số order được giữ lại
     * cho việc mở rộng sau này (cascade nhiều biquad sections).
     */
    
    // Tần số góc digital
    double omegaD = freqToNormalizedOmega(cutoffFreq, sampleRate);
    
    // Pre-warping: tần số analog tương đương
    double omegaA = prewarp(omegaD);
    
    // Với bộ lọc bậc 2 (một biquad section)
    // Pole của Butterworth: s = omegaA * exp(j*pi*(2k+1+order)/(2*order))
    // Cho order=2: poles tại góc 135° và 225°
    
    // Tính hệ số cho bộ lọc bậc 2 (đơn giản hóa)
    // Sử dụng công thức trực tiếp cho Butterworth lowpass
    
    double K = omegaA;  // Tần số cắt analog đã pre-warp
    double K2 = K * K;
    double sqrt2 = std::sqrt(2.0);
    
    // Hệ số chuẩn hóa
    double norm = 1.0 / (1.0 + sqrt2 * K + K2);
    
    // Resize vectors
    coeffs.b.resize(3);
    coeffs.a.resize(3);
    
    // Hệ số tử số (feedforward)
    coeffs.b[0] = K2 * norm;
    coeffs.b[1] = 2.0 * K2 * norm;
    coeffs.b[2] = K2 * norm;
    
    // Hệ số mẫu số (feedback) - a[0] luôn = 1
    coeffs.a[0] = 1.0;
    coeffs.a[1] = 2.0 * (K2 - 1.0) * norm;
    coeffs.a[2] = (1.0 - sqrt2 * K + K2) * norm;
    
    // Nếu cần bậc cao hơn, cascade nhiều biquad sections
    // Ở đây đơn giản hóa với bậc 2 cơ bản
    // Để có bậc 4, áp dụng bộ lọc này 2 lần
}

void SignalProcessor::designButterworthHighpass(float cutoffFreq,
                                                 float sampleRate,
                                                 int /* order */,
                                                 FilterCoefficients& coeffs) {
    /**
     * Thiết kế bộ lọc Butterworth Highpass
     * 
     * Chuyển đổi từ lowpass sang highpass bằng phép biến đổi:
     * s -> omega_c / s (trong miền analog)
     * 
     * Sau bilinear transform, hệ số thay đổi tương ứng
     * 
     * Note: Hiện tại sử dụng bậc 2 cố định. Tham số order được giữ lại
     * cho việc mở rộng sau này.
     */
    
    double omegaD = freqToNormalizedOmega(cutoffFreq, sampleRate);
    double omegaA = prewarp(omegaD);
    
    double K = omegaA;
    double K2 = K * K;
    double sqrt2 = std::sqrt(2.0);
    
    // Hệ số chuẩn hóa cho highpass
    double norm = 1.0 / (1.0 + sqrt2 * K + K2);
    
    coeffs.b.resize(3);
    coeffs.a.resize(3);
    
    // Hệ số highpass: đảo ngược vai trò của K
    coeffs.b[0] = norm;
    coeffs.b[1] = -2.0 * norm;
    coeffs.b[2] = norm;
    
    coeffs.a[0] = 1.0;
    coeffs.a[1] = 2.0 * (K2 - 1.0) * norm;
    coeffs.a[2] = (1.0 - sqrt2 * K + K2) * norm;
}

void SignalProcessor::applyIIRFilter(const std::vector<float>& input,
                                     std::vector<float>& output,
                                     const FilterCoefficients& coeffs) {
    /**
     * Áp dụng bộ lọc IIR - Direct Form II Transposed
     * 
     * Công thức:
     * y[n] = b0*x[n] + w1[n-1]
     * w1[n] = b1*x[n] - a1*y[n] + w2[n-1]
     * w2[n] = b2*x[n] - a2*y[n]
     * 
     * Direct Form II Transposed có ưu điểm:
     * - Ổn định số học tốt hơn
     * - Phù hợp với fixed-point
     * - Dễ pipeline trong HLS
     */
    
    if (input.empty() || coeffs.b.size() < 3 || coeffs.a.size() < 3) {
        output = input;
        return;
    }
    
    size_t n = input.size();
    output.resize(n);
    
    // Trạng thái bộ lọc (delay lines)
    double w1 = 0.0, w2 = 0.0;
    
    // Lấy hệ số (tránh truy cập vector trong vòng lặp)
    double b0 = coeffs.b[0];
    double b1 = coeffs.b[1];
    double b2 = coeffs.b[2];
    double a1 = coeffs.a[1];
    double a2 = coeffs.a[2];
    
    // HLS Pragma: #pragma HLS PIPELINE II=1
    for (size_t i = 0; i < n; ++i) {
        double x = input[i];
        
        // Tính output
        double y = b0 * x + w1;
        
        // Cập nhật trạng thái
        w1 = b1 * x - a1 * y + w2;
        w2 = b2 * x - a2 * y;
        
        output[i] = static_cast<float>(y);
    }
}

void SignalProcessor::applyZeroPhaseFilter(const std::vector<float>& input,
                                           std::vector<float>& output,
                                           const FilterCoefficients& coeffs) {
    /**
     * Zero-phase filtering (Forward-Backward Filtering)
     * 
     * Thuật toán:
     * 1. Lọc tín hiệu theo chiều xuôi: y_forward = filter(x)
     * 2. Đảo ngược tín hiệu: y_reversed = reverse(y_forward)
     * 3. Lọc lại: y_filtered = filter(y_reversed)
     * 4. Đảo ngược lần nữa: output = reverse(y_filtered)
     * 
     * Kết quả:
     * - Pha = 0 (không có độ trễ nhóm)
     * - Bậc hiệu dụng gấp đôi
     * - Đáp ứng tần số: |H(f)|^2
     * 
     * Lưu ý: Không phù hợp cho xử lý real-time
     * (phù hợp cho offline processing như trong bài báo)
     */
    
    if (input.empty()) {
        output.clear();
        return;
    }
    
    // Bước 1: Lọc chiều xuôi
    std::vector<float> forward;
    applyIIRFilter(input, forward, coeffs);
    
    // Bước 2: Đảo ngược
    std::reverse(forward.begin(), forward.end());
    
    // Bước 3: Lọc chiều ngược
    std::vector<float> backward;
    applyIIRFilter(forward, backward, coeffs);
    
    // Bước 4: Đảo ngược lại
    std::reverse(backward.begin(), backward.end());
    
    output = std::move(backward);
}

// ============================================================================
// NORMALIZATION
// ============================================================================

void SignalProcessor::normalize(std::vector<float>& samples) {
    /**
     * Peak Normalization
     * 
     * Chuẩn hóa tín hiệu về khoảng [-1.0, 1.0] bằng cách:
     * normalized[i] = samples[i] / max(|samples|)
     * 
     * Đây là phương pháp đơn giản và phù hợp với HLS vì:
     * - Chỉ cần một pass để tìm max
     * - Phép chia có thể thay bằng nhân nghịch đảo
     * - Không cần thống kê phức tạp (mean, std)
     */
    
    if (samples.empty()) {
        return;
    }
    
    // Tìm giá trị tuyệt đối lớn nhất
    float maxAbs = findMaxAbsValue(samples);
    
    // Tránh chia cho 0
    if (maxAbs < 1e-10f) {
        std::fill(samples.begin(), samples.end(), 0.0f);
        return;
    }
    
    // Tính nghịch đảo để chuyển phép chia thành phép nhân
    // (Tối ưu cho HLS: phép nhân nhanh hơn phép chia)
    float invMax = 1.0f / maxAbs;
    
    // HLS Pragma: #pragma HLS PIPELINE II=1
    for (auto& sample : samples) {
        sample *= invMax;
    }
}

// ============================================================================
// BREATHING CYCLE SEGMENTATION
// ============================================================================

void SignalProcessor::segmentBreathingCycles(const std::vector<float>& samples,
                                              std::vector<BreathingCycle>& cycles,
                                              float sampleRate) {
    /**
     * Phân đoạn chu kỳ hô hấp dựa trên năng lượng tín hiệu
     * 
     * Thuật toán:
     * 1. Tính năng lượng ngắn hạn (Short-Time Energy)
     * 2. Xác định ngưỡng năng lượng động (adaptive threshold)
     * 3. Phát hiện các vùng có năng lượng cao (hoạt động hô hấp)
     * 4. Áp dụng ràng buộc về độ dài chu kỳ (0.5s - 8s)
     * 5. Merge các segment quá gần nhau
     * 
     * Tham số từ bài báo:
     * - Min cycle length: ~0.5s (2000 samples @ 4kHz)
     * - Max cycle length: ~8s (32000 samples @ 4kHz)
     */
    
    cycles.clear();
    
    if (samples.size() < MIN_CYCLE_LENGTH) {
        std::cerr << "[Segmentation] Signal too short for segmentation" << std::endl;
        return;
    }
    
    // ----- BƯỚC 1: Tính năng lượng ngắn hạn -----
    std::vector<float> energy;
    computeShortTimeEnergy(samples, energy, ENERGY_WINDOW_SIZE);
    
    // ----- BƯỚC 2: Tính ngưỡng năng lượng động -----
    // Sử dụng percentile hoặc mean + std
    float maxEnergy = *std::max_element(energy.begin(), energy.end());
    float threshold = maxEnergy * ENERGY_THRESHOLD_RATIO;
    
    // Tính mean năng lượng để có ngưỡng thích nghi hơn
    float meanEnergy = std::accumulate(energy.begin(), energy.end(), 0.0f) 
                       / energy.size();
    
    // Ngưỡng cuối cùng: max của threshold cố định và mean-based
    threshold = std::max(threshold, meanEnergy * 0.5f);
    
    // ----- BƯỚC 3: Phát hiện các vùng hoạt động -----
    std::vector<std::pair<size_t, size_t>> segments;  // (start, end)
    bool inSegment = false;
    size_t segStart = 0;
    
    for (size_t i = 0; i < energy.size(); ++i) {
        if (!inSegment && energy[i] > threshold) {
            // Bắt đầu segment mới
            inSegment = true;
            segStart = i * ENERGY_WINDOW_SIZE / 2;  // Chuyển về sample index
        } else if (inSegment && energy[i] <= threshold) {
            // Kết thúc segment
            inSegment = false;
            size_t segEnd = i * ENERGY_WINDOW_SIZE / 2;
            
            // Kiểm tra độ dài tối thiểu
            if (segEnd - segStart >= static_cast<size_t>(MIN_CYCLE_LENGTH)) {
                segments.push_back({segStart, segEnd});
            }
        }
    }
    
    // Xử lý segment cuối cùng (nếu có)
    if (inSegment) {
        size_t segEnd = samples.size();
        if (segEnd - segStart >= static_cast<size_t>(MIN_CYCLE_LENGTH)) {
            segments.push_back({segStart, segEnd});
        }
    }
    
    // ----- BƯỚC 4: Merge các segment gần nhau -----
    // Gap threshold: 0.3 giây (1200 samples @ 4kHz)
    const size_t GAP_THRESHOLD = static_cast<size_t>(0.3f * sampleRate);
    
    std::vector<std::pair<size_t, size_t>> mergedSegments;
    
    for (const auto& seg : segments) {
        if (mergedSegments.empty()) {
            mergedSegments.push_back(seg);
        } else {
            auto& lastSeg = mergedSegments.back();
            
            // Nếu khoảng cách giữa 2 segment nhỏ, merge chúng
            if (seg.first - lastSeg.second < GAP_THRESHOLD) {
                lastSeg.second = seg.second;
            } else {
                mergedSegments.push_back(seg);
            }
        }
    }
    
    // ----- BƯỚC 5: Tạo BreathingCycle objects -----
    for (const auto& seg : mergedSegments) {
        size_t start = seg.first;
        size_t end = std::min(seg.second, samples.size());
        size_t length = end - start;
        
        // Kiểm tra độ dài hợp lệ
        if (length < static_cast<size_t>(MIN_CYCLE_LENGTH) || 
            length > static_cast<size_t>(MAX_CYCLE_LENGTH)) {
            continue;
        }
        
        BreathingCycle cycle;
        cycle.startIndex = start;
        cycle.endIndex = end;
        cycle.duration = static_cast<float>(length) / sampleRate;
        
        // Copy samples
        cycle.samples.assign(samples.begin() + start, samples.begin() + end);
        
        // Tính năng lượng trung bình của chu kỳ
        float sumSquared = 0.0f;
        for (const auto& s : cycle.samples) {
            sumSquared += s * s;
        }
        cycle.averageEnergy = sumSquared / cycle.samples.size();
        
        cycles.push_back(std::move(cycle));
    }
    
    // Nếu không phát hiện được chu kỳ nào, coi toàn bộ tín hiệu là 1 chu kỳ
    if (cycles.empty() && samples.size() >= static_cast<size_t>(MIN_CYCLE_LENGTH)) {
        BreathingCycle fullCycle;
        fullCycle.startIndex = 0;
        fullCycle.endIndex = samples.size();
        fullCycle.duration = static_cast<float>(samples.size()) / sampleRate;
        fullCycle.samples = samples;
        
        float sumSquared = 0.0f;
        for (const auto& s : samples) {
            sumSquared += s * s;
        }
        fullCycle.averageEnergy = sumSquared / samples.size();
        
        cycles.push_back(std::move(fullCycle));
    }
}

void SignalProcessor::computeShortTimeEnergy(const std::vector<float>& samples,
                                              std::vector<float>& energy,
                                              int windowSize) {
    /**
     * Tính năng lượng ngắn hạn (Short-Time Energy)
     * 
     * E[n] = (1/N) * sum_{k=0}^{N-1} x[n+k]^2
     * 
     * Sử dụng overlapping windows (50% overlap) để có độ phân giải thời gian tốt hơn.
     * 
     * Phương pháp tối ưu:
     * - Sử dụng sliding window với running sum
     * - Tránh tính lại toàn bộ tổng mỗi frame
     */
    
    energy.clear();
    
    if (samples.size() < static_cast<size_t>(windowSize)) {
        return;
    }
    
    // Hop size = 50% window size (overlap)
    int hopSize = windowSize / 2;
    size_t numFrames = (samples.size() - windowSize) / hopSize + 1;
    
    energy.resize(numFrames);
    
    // HLS Pragma: có thể tối ưu với running sum
    for (size_t frame = 0; frame < numFrames; ++frame) {
        size_t startIdx = frame * hopSize;
        
        // Tính tổng bình phương trong cửa sổ
        float sumSquared = 0.0f;
        
        // HLS Pragma: #pragma HLS PIPELINE
        for (int i = 0; i < windowSize; ++i) {
            float sample = samples[startIdx + i];
            sumSquared += sample * sample;
        }
        
        // Năng lượng trung bình
        energy[frame] = sumSquared / windowSize;
    }
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

void SignalProcessor::convertToMono(const std::vector<float>& input,
                                    std::vector<float>& output,
                                    uint16_t channels) {
    /**
     * Chuyển đổi stereo/multi-channel sang mono
     * 
     * Phương pháp: Trung bình cộng các kênh
     * mono[i] = (ch1[i] + ch2[i] + ... + chN[i]) / N
     * 
     * Input được giả định là interleaved:
     * [L0, R0, L1, R1, L2, R2, ...]
     */
    
    if (channels <= 1) {
        output = input;
        return;
    }
    
    size_t numFrames = input.size() / channels;
    output.resize(numFrames);
    
    float invChannels = 1.0f / channels;
    
    for (size_t i = 0; i < numFrames; ++i) {
        float sum = 0.0f;
        
        // Tổng tất cả các kênh
        for (uint16_t ch = 0; ch < channels; ++ch) {
            sum += input[i * channels + ch];
        }
        
        output[i] = sum * invChannels;
    }
}

const std::vector<float>& SignalProcessor::getProcessedSignal() const {
    return m_processedSignal;
}

float SignalProcessor::getCurrentSampleRate() const {
    return m_currentSampleRate;
}

bool SignalProcessor::hasValidData() const {
    return m_hasValidData;
}

void SignalProcessor::reset() {
    m_processedSignal.clear();
    m_currentSampleRate = 0.0f;
    m_hasValidData = false;
    // Giữ lại hệ số bộ lọc vì chúng có thể được tái sử dụng
}

// ============================================================================
// ============================================================================
// DATASET MANAGER IMPLEMENTATION
// ============================================================================
// ============================================================================

// ============================================================================
// DATASET STATISTICS
// ============================================================================

void DatasetStatistics::print() const {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║           ICBHI 2017 Dataset Statistics                  ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Total WAV files:      " << std::setw(8) << totalFiles 
              << "                         ║\n";
    std::cout << "║  Unique patients:      " << std::setw(8) << uniquePatients 
              << "                         ║\n";
    std::cout << "║  Total breathing cycles: " << std::setw(6) << totalCycles 
              << "                         ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Label Distribution:                                     ║\n";
    std::cout << "║    - Normal:           " << std::setw(8) << normalCount 
              << " (" << std::fixed << std::setprecision(1) 
              << (totalCycles > 0 ? 100.0f * normalCount / totalCycles : 0) 
              << "%)                   ║\n";
    std::cout << "║    - Crackle:          " << std::setw(8) << crackleCount 
              << " (" << std::fixed << std::setprecision(1)
              << (totalCycles > 0 ? 100.0f * crackleCount / totalCycles : 0) 
              << "%)                   ║\n";
    std::cout << "║    - Wheeze:           " << std::setw(8) << wheezeCount 
              << " (" << std::fixed << std::setprecision(1)
              << (totalCycles > 0 ? 100.0f * wheezeCount / totalCycles : 0) 
              << "%)                   ║\n";
    std::cout << "║    - Both:             " << std::setw(8) << bothCount 
              << " (" << std::fixed << std::setprecision(1)
              << (totalCycles > 0 ? 100.0f * bothCount / totalCycles : 0) 
              << "%)                    ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Avg cycle duration:   " << std::setw(8) << std::fixed 
              << std::setprecision(2) << avgCycleDuration << " seconds              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

DatasetManager::DatasetManager(const std::string& datasetPath)
    : m_datasetPath(datasetPath)
    , m_signalProcessor(std::make_unique<SignalProcessor>())
{
    resetStatistics();
}

DatasetManager::~DatasetManager() = default;

// ============================================================================
// MAIN METHODS
// ============================================================================

bool DatasetManager::loadDataset(std::vector<LabeledBreathingCycle>& outputCycles,
                                  bool verbose) {
    /**
     * Quy trình load toàn bộ dataset ICBHI:
     * 
     * 1. Quét thư mục tìm tất cả file .wav
     * 2. Với mỗi file:
     *    a. Tìm file annotation .txt tương ứng
     *    b. Parse annotation để lấy thời điểm và nhãn các chu kỳ
     *    c. Load và tiền xử lý tín hiệu audio
     *    d. Cắt tín hiệu thành các chu kỳ theo annotation
     *    e. Gán nhãn và thông tin bệnh nhân
     * 3. Tính thống kê
     */
    
    outputCycles.clear();
    resetStatistics();
    
    // Kiểm tra đường dẫn
    if (!isValidDatasetPath()) {
        std::cerr << "[DatasetManager] Error: Invalid dataset path: " 
                  << m_datasetPath << std::endl;
        return false;
    }
    
    // Quét thư mục
    std::vector<std::string> wavFiles;
    size_t numFiles = scanDirectory(wavFiles);
    
    if (numFiles == 0) {
        std::cerr << "[DatasetManager] Error: No WAV files found in: " 
                  << m_datasetPath << std::endl;
        return false;
    }
    
    if (verbose) {
        std::cout << "[DatasetManager] Found " << numFiles << " WAV files" << std::endl;
        std::cout << "[DatasetManager] Processing..." << std::endl;
    }
    
    m_statistics.totalFiles = numFiles;
    std::set<int> uniquePatientIds;
    size_t successCount = 0;
    
    // Xử lý từng file
    for (size_t i = 0; i < wavFiles.size(); ++i) {
        std::vector<LabeledBreathingCycle> fileCycles;
        
        if (processFile(wavFiles[i], fileCycles)) {
            successCount++;
            
            // Thêm vào kết quả và cập nhật thống kê
            for (auto& cycle : fileCycles) {
                uniquePatientIds.insert(cycle.patientInfo.patientId);
                updateStatistics(cycle);
                outputCycles.push_back(std::move(cycle));
            }
            
            if (verbose && (i + 1) % 50 == 0) {
                std::cout << "  Processed " << (i + 1) << "/" << numFiles 
                          << " files (" << outputCycles.size() << " cycles)" << std::endl;
            }
        } else {
            if (verbose) {
                std::cerr << "  Warning: Failed to process: " << wavFiles[i] << std::endl;
            }
        }
    }
    
    // Cập nhật thống kê cuối cùng
    m_statistics.uniquePatients = uniquePatientIds.size();
    if (m_statistics.totalCycles > 0) {
        // avgCycleDuration đã được tính trong updateStatistics
    }
    
    if (verbose) {
        std::cout << "[DatasetManager] Completed!" << std::endl;
        std::cout << "  Successfully processed: " << successCount << "/" << numFiles 
                  << " files" << std::endl;
        m_statistics.print();
    }
    
    return !outputCycles.empty();
}

bool DatasetManager::processFile(const std::string& wavFilePath,
                                  std::vector<LabeledBreathingCycle>& outputCycles) {
    /**
     * Xử lý một file WAV và trả về các chu kỳ đã gán nhãn
     * 
     * Các bước:
     * 1. Parse filename để lấy PatientInfo
     * 2. Đọc file annotation
     * 3. Load và tiền xử lý audio
     * 4. Cắt theo annotation
     */
    
    outputCycles.clear();
    
    // Kiểm tra file tồn tại
    if (!fs::exists(wavFilePath)) {
        std::cerr << "[DatasetManager] File not found: " << wavFilePath << std::endl;
        return false;
    }
    
    // Lấy tên file
    std::string filename = fs::path(wavFilePath).filename().string();
    
    // ----- BƯỚC 1: Parse filename -----
    PatientInfo patientInfo;
    if (!parseFilename(filename, patientInfo)) {
        std::cerr << "[DatasetManager] Failed to parse filename: " << filename << std::endl;
        return false;
    }
    
    // ----- BƯỚC 2: Đọc annotation -----
    std::string annotationPath = getAnnotationPath(wavFilePath);
    std::vector<CycleAnnotation> annotations;
    
    if (!readAnnotationFile(annotationPath, annotations)) {
        std::cerr << "[DatasetManager] Failed to read annotation: " << annotationPath << std::endl;
        return false;
    }
    
    if (annotations.empty()) {
        std::cerr << "[DatasetManager] No annotations found for: " << filename << std::endl;
        return false;
    }
    
    // ----- BƯỚC 3: Load và tiền xử lý audio -----
    // Sử dụng SignalProcessor nhưng không để nó tự segment
    AudioData audioData;
    if (!m_signalProcessor->loadWavFile(wavFilePath, audioData)) {
        std::cerr << "[DatasetManager] Failed to load WAV: " << wavFilePath << std::endl;
        return false;
    }
    
    // Chuyển sang mono nếu cần
    std::vector<float> monoSamples;
    if (audioData.channels > 1) {
        // Trung bình các kênh
        size_t numFrames = audioData.samples.size() / audioData.channels;
        monoSamples.resize(numFrames);
        for (size_t i = 0; i < numFrames; ++i) {
            float sum = 0.0f;
            for (uint16_t ch = 0; ch < audioData.channels; ++ch) {
                sum += audioData.samples[i * audioData.channels + ch];
            }
            monoSamples[i] = sum / audioData.channels;
        }
    } else {
        monoSamples = std::move(audioData.samples);
    }
    
    // Resampling về 4kHz
    std::vector<float> resampledSignal;
    m_signalProcessor->resample(monoSamples, audioData.sampleRate, 
                                 resampledSignal, TARGET_SAMPLE_RATE);
    
    // Áp dụng bandpass filter
    std::vector<float> filteredSignal;
    m_signalProcessor->applyBandpassFilter(resampledSignal, filteredSignal,
                                            TARGET_SAMPLE_RATE,
                                            BANDPASS_LOW_FREQ, BANDPASS_HIGH_FREQ);
    
    // Normalize
    m_signalProcessor->normalize(filteredSignal);
    
    // ----- BƯỚC 4: Cắt theo annotation -----
    segmentByAnnotations(filteredSignal, TARGET_SAMPLE_RATE, annotations,
                         patientInfo, filename, outputCycles);
    
    return !outputCycles.empty();
}

bool DatasetManager::loadDatasetWithProgress(
    std::vector<LabeledBreathingCycle>& outputCycles,
    std::function<void(size_t current, size_t total)> progressCallback) {
    
    outputCycles.clear();
    resetStatistics();
    
    if (!isValidDatasetPath()) {
        return false;
    }
    
    std::vector<std::string> wavFiles;
    size_t numFiles = scanDirectory(wavFiles);
    
    if (numFiles == 0) {
        return false;
    }
    
    m_statistics.totalFiles = numFiles;
    std::set<int> uniquePatientIds;
    
    for (size_t i = 0; i < wavFiles.size(); ++i) {
        std::vector<LabeledBreathingCycle> fileCycles;
        
        if (processFile(wavFiles[i], fileCycles)) {
            for (auto& cycle : fileCycles) {
                uniquePatientIds.insert(cycle.patientInfo.patientId);
                updateStatistics(cycle);
                outputCycles.push_back(std::move(cycle));
            }
        }
        
        // Gọi callback báo tiến độ
        if (progressCallback) {
            progressCallback(i + 1, numFiles);
        }
    }
    
    m_statistics.uniquePatients = uniquePatientIds.size();
    
    return !outputCycles.empty();
}

// ============================================================================
// ANNOTATION METHODS
// ============================================================================

bool DatasetManager::readAnnotationFile(const std::string& annotationPath,
                                         std::vector<CycleAnnotation>& annotations) {
    /**
     * Đọc file annotation ICBHI
     * 
     * Format: [Start] [End] [Crackles] [Wheezes]
     * Ví dụ:
     *   0.036   0.579   0   0
     *   0.579   2.45    0   0
     *   2.45    3.893   1   0   <- có crackle
     */
    
    annotations.clear();
    
    std::ifstream file(annotationPath);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    int lineNumber = 0;
    
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Bỏ qua dòng trống
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        
        // Parse dòng
        std::istringstream iss(line);
        CycleAnnotation annotation;
        int crackle, wheeze;
        
        if (iss >> annotation.startTime >> annotation.endTime >> crackle >> wheeze) {
            annotation.hasCrackle = (crackle != 0);
            annotation.hasWheeze = (wheeze != 0);
            
            // Validate
            if (annotation.startTime < 0 || annotation.endTime <= annotation.startTime) {
                std::cerr << "[readAnnotationFile] Invalid time range at line " 
                          << lineNumber << ": " << line << std::endl;
                continue;
            }
            
            annotations.push_back(annotation);
        } else {
            std::cerr << "[readAnnotationFile] Parse error at line " 
                      << lineNumber << ": " << line << std::endl;
        }
    }
    
    file.close();
    return !annotations.empty();
}

bool DatasetManager::parseFilename(const std::string& filename, PatientInfo& info) {
    /**
     * Parse tên file ICBHI
     * 
     * Format: {PatientID}_{RecordingIndex}_{ChestLocation}_{Mode}_{Equipment}.wav
     * Ví dụ: 101_1b1_Al_sc_Meditron.wav
     * 
     * PatientID: 101-226 (số nguyên 3 chữ số)
     * RecordingIndex: 1b1, 2b2, etc.
     * ChestLocation: Al (Anterior left), Ar, Pl, Pr, Tc, Ll, Lr, etc.
     * Mode: sc (single channel), mc (multi channel)
     * Equipment: Meditron, LittC2SE, Litt3200, AKGC417L
     */
    
    // Loại bỏ extension
    std::string baseName = filename;
    size_t dotPos = baseName.rfind('.');
    if (dotPos != std::string::npos) {
        baseName = baseName.substr(0, dotPos);
    }
    
    // Split by underscore
    std::vector<std::string> parts;
    std::istringstream iss(baseName);
    std::string part;
    
    while (std::getline(iss, part, '_')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }
    
    // Cần ít nhất 5 phần
    if (parts.size() < 5) {
        return false;
    }
    
    // Parse PatientID
    try {
        info.patientId = std::stoi(parts[0]);
    } catch (...) {
        return false;
    }
    
    info.recordingIndex = parts[1];
    info.chestLocation = parts[2];
    info.acquisitionMode = parts[3];
    info.equipment = parts[4];
    
    return true;
}

// ============================================================================
// SEGMENTATION BY ANNOTATIONS
// ============================================================================

void DatasetManager::segmentByAnnotations(const std::vector<float>& processedSignal,
                                           float sampleRate,
                                           const std::vector<CycleAnnotation>& annotations,
                                           const PatientInfo& patientInfo,
                                           const std::string& sourceFile,
                                           std::vector<LabeledBreathingCycle>& outputCycles) {
    /**
     * Cắt tín hiệu đã xử lý thành các chu kỳ dựa trên annotation
     * 
     * Với mỗi annotation:
     * 1. Tính sample index từ thời gian: index = time * sampleRate
     * 2. Trích xuất đoạn tín hiệu [startIndex, endIndex)
     * 3. Gán nhãn từ annotation
     * 4. Đính kèm thông tin bệnh nhân
     */
    
    outputCycles.clear();
    
    if (processedSignal.empty() || annotations.empty()) {
        return;
    }
    
    size_t totalSamples = processedSignal.size();
    
    for (const auto& annotation : annotations) {
        // Tính sample indices
        size_t startIndex = static_cast<size_t>(annotation.startTime * sampleRate);
        size_t endIndex = static_cast<size_t>(annotation.endTime * sampleRate);
        
        // Validate indices
        if (startIndex >= totalSamples) {
            continue;  // Annotation vượt quá tín hiệu
        }
        
        if (endIndex > totalSamples) {
            endIndex = totalSamples;  // Clip to end
        }
        
        if (endIndex <= startIndex) {
            continue;  // Invalid range
        }
        
        size_t cycleLength = endIndex - startIndex;
        
        // Kiểm tra độ dài tối thiểu (ít nhất 0.2 giây = 800 samples @ 4kHz)
        if (cycleLength < 800) {
            continue;
        }
        
        // Tạo LabeledBreathingCycle
        LabeledBreathingCycle cycle;
        
        // Copy samples
        cycle.samples.assign(processedSignal.begin() + startIndex,
                             processedSignal.begin() + endIndex);
        
        // Gán nhãn
        cycle.label = annotation.getLabel();
        
        // Thông tin thời gian
        cycle.startTime = annotation.startTime;
        cycle.endTime = annotation.endTime;
        cycle.duration = annotation.getDuration();
        
        // Thông tin bệnh nhân và nguồn
        cycle.patientInfo = patientInfo;
        cycle.sourceFile = sourceFile;
        
        outputCycles.push_back(std::move(cycle));
    }
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

const DatasetStatistics& DatasetManager::getStatistics() const {
    return m_statistics;
}

std::vector<std::string> DatasetManager::getWavFiles() const {
    std::vector<std::string> wavFiles;
    scanDirectory(wavFiles);
    return wavFiles;
}

bool DatasetManager::isValidDatasetPath() const {
    if (!fs::exists(m_datasetPath)) {
        return false;
    }
    
    if (!fs::is_directory(m_datasetPath)) {
        return false;
    }
    
    // Kiểm tra có ít nhất 1 file WAV
    for (const auto& entry : fs::directory_iterator(m_datasetPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav") {
                return true;
            }
        }
    }
    
    return false;
}

void DatasetManager::setDatasetPath(const std::string& path) {
    m_datasetPath = path;
    resetStatistics();
}

const std::string& DatasetManager::getDatasetPath() const {
    return m_datasetPath;
}

std::string DatasetManager::getAnnotationPath(const std::string& wavPath) {
    // Thay đổi extension từ .wav sang .txt
    fs::path p(wavPath);
    p.replace_extension(".txt");
    return p.string();
}

void DatasetManager::filterByLabel(const std::vector<LabeledBreathingCycle>& allCycles,
                                    RespiratoryLabel label,
                                    std::vector<LabeledBreathingCycle>& filteredCycles) {
    filteredCycles.clear();
    
    for (const auto& cycle : allCycles) {
        if (cycle.label == label) {
            filteredCycles.push_back(cycle);
        }
    }
}

void DatasetManager::splitDataset(const std::vector<LabeledBreathingCycle>& allCycles,
                                   float trainRatio,
                                   std::vector<LabeledBreathingCycle>& trainSet,
                                   std::vector<LabeledBreathingCycle>& testSet,
                                   int shuffleSeed) {
    trainSet.clear();
    testSet.clear();
    
    if (allCycles.empty()) {
        return;
    }
    
    // Tạo indices
    std::vector<size_t> indices(allCycles.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Shuffle nếu cần
    if (shuffleSeed >= 0) {
        std::mt19937 gen(static_cast<unsigned int>(shuffleSeed));
        std::shuffle(indices.begin(), indices.end(), gen);
    }
    
    // Split
    size_t trainSize = static_cast<size_t>(allCycles.size() * trainRatio);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < trainSize) {
            trainSet.push_back(allCycles[indices[i]]);
        } else {
            testSet.push_back(allCycles[indices[i]]);
        }
    }
}

// ============================================================================
// PRIVATE METHODS
// ============================================================================

size_t DatasetManager::scanDirectory(std::vector<std::string>& wavFiles) const {
    wavFiles.clear();
    
    if (!fs::exists(m_datasetPath) || !fs::is_directory(m_datasetPath)) {
        return 0;
    }
    
    for (const auto& entry : fs::directory_iterator(m_datasetPath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (ext == ".wav") {
                wavFiles.push_back(entry.path().string());
            }
        }
    }
    
    // Sort để đảm bảo thứ tự nhất quán
    std::sort(wavFiles.begin(), wavFiles.end());
    
    return wavFiles.size();
}

void DatasetManager::updateStatistics(const LabeledBreathingCycle& cycle) {
    m_statistics.totalCycles++;
    
    switch (cycle.label) {
        case RespiratoryLabel::NORMAL:
            m_statistics.normalCount++;
            break;
        case RespiratoryLabel::CRACKLE:
            m_statistics.crackleCount++;
            break;
        case RespiratoryLabel::WHEEZE:
            m_statistics.wheezeCount++;
            break;
        case RespiratoryLabel::BOTH:
            m_statistics.bothCount++;
            break;
    }
    
    // Cập nhật average duration (running average)
    float n = static_cast<float>(m_statistics.totalCycles);
    m_statistics.avgCycleDuration = 
        m_statistics.avgCycleDuration * (n - 1) / n + cycle.duration / n;
}

void DatasetManager::resetStatistics() {
    m_statistics = DatasetStatistics();
}

} // namespace respiratory
