/**
 * @file FeatureExtraction.cpp
 * @brief Implementation of Feature Extraction Module
 * 
 * Triển khai chi tiết các hàm trích xuất đặc trưng cho hệ thống
 * phân tích âm thanh hô hấp theo bài báo IEEE.
 * 
 * Features:
 * - Time-Domain: EED, ZCR, RMSE
 * - Frequency-Domain: MFCC (39-dimensional)
 * 
 * @author Research Team
 * @date 2026
 */

#include "FeatureExtraction.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>

// Kiểm tra và sử dụng FFTW3 nếu có
#ifdef USE_FFTW3
#include <fftw3.h>
#endif

namespace respiratory {

// ============================================================================
// CONSTANTS
// ============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// CycleFeatures IMPLEMENTATION
// ============================================================================

std::vector<float> CycleFeatures::toFlatVector() const {
    /**
     * Format vector phẳng:
     * [EED, ZCR_mean, ZCR_std, RMSE_mean, RMSE_std,    (5)
     *  MFCC_mean(13), MFCC_std(13),                    (26)
     *  Delta_mean(13), Delta_std(13),                  (26)
     *  Delta2_mean(13), Delta2_std(13)]                (26)
     * 
     * Total: 5 + 26 + 26 + 26 = 83 features
     */
    
    std::vector<float> flat;
    flat.reserve(83);
    
    // Time-domain features
    flat.push_back(eed);
    flat.push_back(zcr_mean);
    flat.push_back(zcr_std);
    flat.push_back(rmse_mean);
    flat.push_back(rmse_std);
    
    // MFCC mean và std
    flat.insert(flat.end(), mfcc_mean.begin(), mfcc_mean.end());
    flat.insert(flat.end(), mfcc_std.begin(), mfcc_std.end());
    
    // Delta mean và std
    flat.insert(flat.end(), delta_mean.begin(), delta_mean.end());
    flat.insert(flat.end(), delta_std.begin(), delta_std.end());
    
    // Delta-Delta mean và std
    flat.insert(flat.end(), delta2_mean.begin(), delta2_mean.end());
    flat.insert(flat.end(), delta2_std.begin(), delta2_std.end());
    
    return flat;
}

int CycleFeatures::getFeatureDimension() {
    return 5 + NUM_MFCC_COEFFS * 6;  // 83
}

std::vector<std::string> CycleFeatures::getFeatureNames() {
    std::vector<std::string> names;
    names.reserve(83);
    
    // Time-domain
    names.push_back("EED");
    names.push_back("ZCR_mean");
    names.push_back("ZCR_std");
    names.push_back("RMSE_mean");
    names.push_back("RMSE_std");
    
    // MFCC
    for (int i = 0; i < NUM_MFCC_COEFFS; ++i) {
        names.push_back("MFCC" + std::to_string(i) + "_mean");
    }
    for (int i = 0; i < NUM_MFCC_COEFFS; ++i) {
        names.push_back("MFCC" + std::to_string(i) + "_std");
    }
    
    // Delta
    for (int i = 0; i < NUM_MFCC_COEFFS; ++i) {
        names.push_back("Delta" + std::to_string(i) + "_mean");
    }
    for (int i = 0; i < NUM_MFCC_COEFFS; ++i) {
        names.push_back("Delta" + std::to_string(i) + "_std");
    }
    
    // Delta-Delta
    for (int i = 0; i < NUM_MFCC_COEFFS; ++i) {
        names.push_back("Delta2_" + std::to_string(i) + "_mean");
    }
    for (int i = 0; i < NUM_MFCC_COEFFS; ++i) {
        names.push_back("Delta2_" + std::to_string(i) + "_std");
    }
    
    return names;
}

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

FeatureExtractor::FeatureExtractor(float sampleRate)
    : m_sampleRate(sampleRate)
    , m_frameSize(static_cast<int>(FRAME_SIZE_MS * sampleRate / 1000.0f))
    , m_hopSize(static_cast<int>(m_frameSize * (1.0f - FRAME_OVERLAP_RATIO)))
    , m_fftSize(FFT_SIZE)
    , m_initialized(false)
{
    // Khởi tạo tất cả lookup tables và buffers
    initHammingWindow();
    initMelFilterbank();
    initDCTMatrix();
    initFFT();
    
    m_initialized = true;
}

FeatureExtractor::~FeatureExtractor() {
    // Cleanup nếu dùng FFTW3
#ifdef USE_FFTW3
    // fftw_cleanup();
#endif
}

// ============================================================================
// INITIALIZATION METHODS
// ============================================================================

void FeatureExtractor::initHammingWindow() {
    /**
     * Hamming Window: w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
     * 
     * Pre-compute để tránh tính toán trong runtime
     */
    
    m_hammingWindow.resize(m_frameSize);
    
    for (int n = 0; n < m_frameSize; ++n) {
        m_hammingWindow[n] = 0.54f - 0.46f * std::cos(2.0f * M_PI * n / (m_frameSize - 1));
    }
}

void FeatureExtractor::initMelFilterbank() {
    /**
     * Khởi tạo Mel Filterbank
     * 
     * 1. Chuyển đổi low/high freq sang Mel scale
     * 2. Tạo NUM_MEL_FILTERS + 2 điểm trên Mel scale
     * 3. Chuyển ngược về Hz
     * 4. Chuyển sang FFT bin indices
     * 5. Tạo triangular filters
     */
    
    // Tính Mel frequencies
    float melLow = hzToMel(MEL_LOW_FREQ);
    float melHigh = hzToMel(MEL_HIGH_FREQ);
    
    // Tạo NUM_MEL_FILTERS + 2 điểm đều trên Mel scale
    std::vector<float> melPoints(NUM_MEL_FILTERS + 2);
    for (int i = 0; i < NUM_MEL_FILTERS + 2; ++i) {
        melPoints[i] = melLow + i * (melHigh - melLow) / (NUM_MEL_FILTERS + 1);
    }
    
    // Chuyển về Hz
    std::vector<float> hzPoints(NUM_MEL_FILTERS + 2);
    for (int i = 0; i < NUM_MEL_FILTERS + 2; ++i) {
        hzPoints[i] = melToHz(melPoints[i]);
    }
    
    // Chuyển sang FFT bin indices
    std::vector<int> binIndices(NUM_MEL_FILTERS + 2);
    for (int i = 0; i < NUM_MEL_FILTERS + 2; ++i) {
        binIndices[i] = static_cast<int>(std::floor((m_fftSize + 1) * hzPoints[i] / m_sampleRate));
    }
    
    // Tạo filterbank matrix
    m_melFilterbank.resize(NUM_MEL_FILTERS);
    m_melFilterStart.resize(NUM_MEL_FILTERS);
    m_melFilterEnd.resize(NUM_MEL_FILTERS);
    
    int spectrumSize = m_fftSize / 2 + 1;
    
    for (int m = 0; m < NUM_MEL_FILTERS; ++m) {
        m_melFilterbank[m].resize(spectrumSize, 0.0f);
        
        int startBin = binIndices[m];
        int centerBin = binIndices[m + 1];
        int endBin = binIndices[m + 2];
        
        m_melFilterStart[m] = startBin;
        m_melFilterEnd[m] = endBin;
        
        // Rising slope
        for (int k = startBin; k < centerBin && k < spectrumSize; ++k) {
            if (centerBin != startBin) {
                m_melFilterbank[m][k] = static_cast<float>(k - startBin) / (centerBin - startBin);
            }
        }
        
        // Falling slope
        for (int k = centerBin; k < endBin && k < spectrumSize; ++k) {
            if (endBin != centerBin) {
                m_melFilterbank[m][k] = static_cast<float>(endBin - k) / (endBin - centerBin);
            }
        }
    }
}

void FeatureExtractor::initDCTMatrix() {
    /**
     * DCT-II matrix cho MFCC
     * 
     * c[k] = sqrt(2/N) * sum_{n=0}^{N-1} x[n] * cos(π*k*(2n+1)/(2N))
     * 
     * Pre-compute ma trận DCT để tránh tính cos trong runtime
     */
    
    m_dctMatrix.resize(NUM_MFCC_COEFFS);
    
    float normFactor = std::sqrt(2.0f / NUM_MEL_FILTERS);
    
    for (int k = 0; k < NUM_MFCC_COEFFS; ++k) {
        m_dctMatrix[k].resize(NUM_MEL_FILTERS);
        
        for (int n = 0; n < NUM_MEL_FILTERS; ++n) {
            m_dctMatrix[k][n] = normFactor * 
                std::cos(M_PI * k * (2.0f * n + 1.0f) / (2.0f * NUM_MEL_FILTERS));
        }
    }
    
    // Normalize first coefficient (k=0)
    for (int n = 0; n < NUM_MEL_FILTERS; ++n) {
        m_dctMatrix[0][n] *= std::sqrt(0.5f);
    }
}

void FeatureExtractor::initFFT() {
    /**
     * Khởi tạo FFT buffers
     * 
     * Nếu có FFTW3, sử dụng nó. Nếu không, dùng built-in DFT.
     */
    
    m_fftInput.resize(m_fftSize, 0.0f);
    m_fftOutput.resize(m_fftSize / 2 + 1);
    
#ifdef USE_FFTW3
    // FFTW3 initialization would go here
    // fftwf_plan_dft_r2c_1d(...)
#endif
}

// ============================================================================
// MAIN EXTRACTION METHODS
// ============================================================================

bool FeatureExtractor::extractFeatures(const std::vector<float>& samples,
                                        CycleFeatures& features,
                                        bool keepFrameFeatures) {
    /**
     * Quy trình trích xuất đặc trưng chính:
     * 
     * 1. Tính time-domain features cho toàn cycle
     * 2. Chia thành frames
     * 3. Với mỗi frame: tính MFCC, ZCR, RMSE
     * 4. Tính Delta và Delta-Delta MFCC
     * 5. Aggregate features (mean, std)
     */
    
    if (!m_initialized) {
        std::cerr << "[FeatureExtractor] Not initialized!" << std::endl;
        return false;
    }
    
    if (samples.size() < static_cast<size_t>(m_frameSize)) {
        std::cerr << "[FeatureExtractor] Signal too short: " << samples.size() 
                  << " samples (need at least " << m_frameSize << ")" << std::endl;
        return false;
    }
    
    // Reset features
    features = CycleFeatures();
    features.durationSec = static_cast<float>(samples.size()) / m_sampleRate;
    
    // ----- BƯỚC 1: Tính EED cho toàn cycle -----
    features.eed = computeEED(samples);
    
    // ----- BƯỚC 2: Chia thành frames và tính features -----
    int numFrames = (static_cast<int>(samples.size()) - m_frameSize) / m_hopSize + 1;
    features.numFrames = numFrames;
    
    if (numFrames < 1) {
        std::cerr << "[FeatureExtractor] Not enough samples for even one frame" << std::endl;
        return false;
    }
    
    // Lưu trữ MFCC của tất cả frames để tính delta
    std::vector<std::vector<float>> allMFCC(numFrames);
    std::vector<float> allZCR(numFrames);
    std::vector<float> allRMSE(numFrames);
    
    if (keepFrameFeatures) {
        features.frameFeatures.resize(numFrames);
    }
    
    // Trích xuất features cho từng frame
    for (int f = 0; f < numFrames; ++f) {
        int startIdx = f * m_hopSize;
        
        // Lấy frame
        std::vector<float> frame(samples.begin() + startIdx,
                                  samples.begin() + startIdx + m_frameSize);
        
        // Tính MFCC
        std::vector<float> mfcc;
        extractFrameMFCC(frame, mfcc);
        allMFCC[f] = mfcc;
        
        // Tính ZCR và RMSE
        float zcr = computeZCR(frame);
        float rmse = computeRMSE(frame);
        allZCR[f] = zcr;
        allRMSE[f] = rmse;
        
        if (keepFrameFeatures) {
            features.frameFeatures[f].mfcc = mfcc;
            features.frameFeatures[f].zcr = zcr;
            features.frameFeatures[f].rmse = rmse;
        }
    }
    
    // ----- BƯỚC 3: Tính Delta và Delta-Delta -----
    std::vector<std::vector<float>> allDelta;
    std::vector<std::vector<float>> allDelta2;
    
    computeDeltaCoeffs(allMFCC, allDelta, DELTA_WINDOW);
    computeDeltaCoeffs(allDelta, allDelta2, DELTA_WINDOW);
    
    // ----- BƯỚC 4: Aggregate features -----
    // ZCR statistics
    float zcrSum = std::accumulate(allZCR.begin(), allZCR.end(), 0.0f);
    features.zcr_mean = zcrSum / numFrames;
    
    float zcrSqSum = 0.0f;
    for (float z : allZCR) {
        zcrSqSum += (z - features.zcr_mean) * (z - features.zcr_mean);
    }
    features.zcr_std = std::sqrt(zcrSqSum / numFrames);
    
    // RMSE statistics
    float rmseSum = std::accumulate(allRMSE.begin(), allRMSE.end(), 0.0f);
    features.rmse_mean = rmseSum / numFrames;
    
    float rmseSqSum = 0.0f;
    for (float r : allRMSE) {
        rmseSqSum += (r - features.rmse_mean) * (r - features.rmse_mean);
    }
    features.rmse_std = std::sqrt(rmseSqSum / numFrames);
    
    // MFCC statistics
    for (int c = 0; c < NUM_MFCC_COEFFS; ++c) {
        // Mean
        float sum = 0.0f;
        for (int f = 0; f < numFrames; ++f) {
            sum += allMFCC[f][c];
        }
        features.mfcc_mean[c] = sum / numFrames;
        
        // Std
        float sqSum = 0.0f;
        for (int f = 0; f < numFrames; ++f) {
            float diff = allMFCC[f][c] - features.mfcc_mean[c];
            sqSum += diff * diff;
        }
        features.mfcc_std[c] = std::sqrt(sqSum / numFrames);
    }
    
    // Delta statistics
    for (int c = 0; c < NUM_MFCC_COEFFS; ++c) {
        float sum = 0.0f;
        for (int f = 0; f < numFrames; ++f) {
            sum += allDelta[f][c];
        }
        features.delta_mean[c] = sum / numFrames;
        
        float sqSum = 0.0f;
        for (int f = 0; f < numFrames; ++f) {
            float diff = allDelta[f][c] - features.delta_mean[c];
            sqSum += diff * diff;
        }
        features.delta_std[c] = std::sqrt(sqSum / numFrames);
    }
    
    // Delta-Delta statistics
    for (int c = 0; c < NUM_MFCC_COEFFS; ++c) {
        float sum = 0.0f;
        for (int f = 0; f < numFrames; ++f) {
            sum += allDelta2[f][c];
        }
        features.delta2_mean[c] = sum / numFrames;
        
        float sqSum = 0.0f;
        for (int f = 0; f < numFrames; ++f) {
            float diff = allDelta2[f][c] - features.delta2_mean[c];
            sqSum += diff * diff;
        }
        features.delta2_std[c] = std::sqrt(sqSum / numFrames);
    }
    
    return true;
}

bool FeatureExtractor::extractFlatFeatures(const std::vector<float>& samples,
                                            std::vector<float>& flatFeatures) {
    CycleFeatures features;
    
    if (!extractFeatures(samples, features, false)) {
        return false;
    }
    
    flatFeatures = features.toFlatVector();
    return true;
}

// ============================================================================
// TIME-DOMAIN FEATURE EXTRACTION
// ============================================================================

float FeatureExtractor::computeEED(const std::vector<float>& samples) {
    /**
     * EED (Extreme Energy Difference)
     * 
     * Công thức gốc: EED = |(w_max^T * x)^2 - (w_min^T * x)^2|
     * 
     * Simplified implementation:
     * Tính energy của các frames, sau đó lấy |E_max - E_min|
     * 
     * Đây là approximation hợp lý vì EED đo sự chênh lệch
     * năng lượng cực đại trong tín hiệu.
     */
    
    if (samples.size() < static_cast<size_t>(m_frameSize)) {
        return 0.0f;
    }
    
    int numFrames = (static_cast<int>(samples.size()) - m_frameSize) / m_hopSize + 1;
    
    float maxEnergy = -std::numeric_limits<float>::max();
    float minEnergy = std::numeric_limits<float>::max();
    
    for (int f = 0; f < numFrames; ++f) {
        int startIdx = f * m_hopSize;
        
        // Tính energy của frame
        float energy = 0.0f;
        for (int i = 0; i < m_frameSize; ++i) {
            float sample = samples[startIdx + i];
            energy += sample * sample;
        }
        energy /= m_frameSize;
        
        if (energy > maxEnergy) maxEnergy = energy;
        if (energy < minEnergy) minEnergy = energy;
    }
    
    return std::abs(maxEnergy - minEnergy);
}

float FeatureExtractor::computeZCR(const std::vector<float>& samples) {
    /**
     * ZCR (Zero Crossing Rate)
     * 
     * Công thức: ZCR = (1 / 2(N-1)) * sum_{n=1}^{N-1} |sgn[x(n)] - sgn[x(n-1)]|
     * 
     * Đếm số lần tín hiệu đi qua 0, chuẩn hóa về [0, 1]
     */
    
    if (samples.size() < 2) {
        return 0.0f;
    }
    
    int N = static_cast<int>(samples.size());
    int crossings = 0;
    
    for (int n = 1; n < N; ++n) {
        int signDiff = std::abs(sign(samples[n]) - sign(samples[n - 1]));
        crossings += signDiff;
    }
    
    // ZCR = crossings / (2 * (N-1))
    // Chia cho 2 vì |sgn[x(n)] - sgn[x(n-1)]| có thể là 0 hoặc 2
    return static_cast<float>(crossings) / (2.0f * (N - 1));
}

float FeatureExtractor::computeRMSE(const std::vector<float>& samples) {
    /**
     * RMSE (Root Mean Square Energy)
     * 
     * Công thức: RMSE = sqrt((1/N) * sum_{n=0}^{N-1} |x(n)|^2)
     */
    
    if (samples.empty()) {
        return 0.0f;
    }
    
    float sumSquared = 0.0f;
    for (float sample : samples) {
        sumSquared += sample * sample;
    }
    
    return std::sqrt(sumSquared / samples.size());
}

// ============================================================================
// MFCC EXTRACTION
// ============================================================================

void FeatureExtractor::extractFrameMFCC(const std::vector<float>& frame,
                                         std::vector<float>& mfcc) {
    /**
     * Quy trình trích xuất MFCC:
     * 
     * 1. Pre-emphasis: y[n] = x[n] - 0.97*x[n-1]
     * 2. Hamming window
     * 3. Zero-padding và FFT
     * 4. Power spectrum: |X[k]|^2
     * 5. Mel filterbank
     * 6. Log compression
     * 7. DCT
     */
    
    mfcc.resize(NUM_MFCC_COEFFS);
    
    // ----- BƯỚC 1: Pre-emphasis -----
    std::vector<float> emphasized;
    applyPreEmphasis(frame, emphasized, PRE_EMPHASIS_COEFF);
    
    // ----- BƯỚC 2: Hamming window -----
    applyHammingWindow(emphasized);
    
    // ----- BƯỚC 3: FFT -----
    std::vector<std::complex<float>> spectrum;
    computeFFT(emphasized, spectrum);
    
    // ----- BƯỚC 4: Power spectrum -----
    std::vector<float> powerSpectrum;
    computePowerSpectrum(spectrum, powerSpectrum);
    
    // ----- BƯỚC 5: Mel filterbank -----
    std::vector<float> melEnergies;
    applyMelFilterbank(powerSpectrum, melEnergies);
    
    // ----- BƯỚC 6: Log compression -----
    for (float& e : melEnergies) {
        // Thêm small constant để tránh log(0)
        e = std::log(e + 1e-10f);
    }
    
    // ----- BƯỚC 7: DCT -----
    computeDCT(melEnergies, mfcc, NUM_MFCC_COEFFS);
}

void FeatureExtractor::computeDeltaCoeffs(const std::vector<std::vector<float>>& coeffs,
                                           std::vector<std::vector<float>>& delta,
                                           int windowSize) {
    /**
     * Tính Delta (velocity) coefficients
     * 
     * Công thức:
     * d[t] = (sum_{n=1}^{N} n * (c[t+n] - c[t-n])) / (2 * sum_{n=1}^{N} n^2)
     * 
     * Với boundary handling: pad với giá trị đầu/cuối
     */
    
    int numFrames = static_cast<int>(coeffs.size());
    int numCoeffs = static_cast<int>(coeffs[0].size());
    
    delta.resize(numFrames);
    for (int t = 0; t < numFrames; ++t) {
        delta[t].resize(numCoeffs, 0.0f);
    }
    
    // Tính mẫu số: 2 * sum_{n=1}^{N} n^2
    float denominator = 0.0f;
    for (int n = 1; n <= windowSize; ++n) {
        denominator += n * n;
    }
    denominator *= 2.0f;
    
    if (denominator < 1e-10f) {
        return;  // Avoid division by zero
    }
    
    for (int t = 0; t < numFrames; ++t) {
        for (int c = 0; c < numCoeffs; ++c) {
            float numerator = 0.0f;
            
            for (int n = 1; n <= windowSize; ++n) {
                // Index với boundary clamping
                int idxPlus = std::min(t + n, numFrames - 1);
                int idxMinus = std::max(t - n, 0);
                
                numerator += n * (coeffs[idxPlus][c] - coeffs[idxMinus][c]);
            }
            
            delta[t][c] = numerator / denominator;
        }
    }
}

// ============================================================================
// DSP HELPER METHODS
// ============================================================================

void FeatureExtractor::applyPreEmphasis(const std::vector<float>& samples,
                                         std::vector<float>& output,
                                         float coeff) {
    /**
     * Pre-emphasis filter: y[n] = x[n] - α * x[n-1]
     * 
     * Mục đích: Boost high frequencies để cân bằng spectrum
     * (vì speech/respiratory sounds có more energy ở low freq)
     */
    
    output.resize(samples.size());
    
    if (samples.empty()) {
        return;
    }
    
    output[0] = samples[0];  // First sample unchanged
    
    for (size_t n = 1; n < samples.size(); ++n) {
        output[n] = samples[n] - coeff * samples[n - 1];
    }
}

void FeatureExtractor::applyHammingWindow(std::vector<float>& samples) {
    /**
     * Áp dụng Hamming window (in-place)
     * 
     * Sử dụng pre-computed window từ m_hammingWindow
     */
    
    int N = std::min(static_cast<int>(samples.size()), m_frameSize);
    
    for (int n = 0; n < N; ++n) {
        samples[n] *= m_hammingWindow[n];
    }
}

void FeatureExtractor::computeFFT(const std::vector<float>& samples,
                                   std::vector<std::complex<float>>& spectrum) {
    /**
     * Tính FFT
     * 
     * Implementation: Cooley-Tukey radix-2 FFT
     * Nếu FFTW3 available, sẽ sử dụng nó thay thế.
     */
    
    int N = m_fftSize;
    spectrum.resize(N / 2 + 1);
    
    // Zero-pad input
    std::vector<std::complex<float>> x(N, std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < std::min(samples.size(), static_cast<size_t>(N)); ++i) {
        x[i] = std::complex<float>(samples[i], 0.0f);
    }
    
#ifdef USE_FFTW3
    // FFTW3 implementation
    // fftwf_execute(plan);
    // Copy results...
#else
    // Built-in DFT (for small N, this is acceptable)
    // Cooley-Tukey radix-2 FFT
    
    // Bit-reversal permutation
    int bits = static_cast<int>(std::log2(N));
    for (int i = 0; i < N; ++i) {
        int j = 0;
        for (int k = 0; k < bits; ++k) {
            if (i & (1 << k)) {
                j |= (1 << (bits - 1 - k));
            }
        }
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }
    
    // FFT butterfly
    for (int size = 2; size <= N; size *= 2) {
        int halfSize = size / 2;
        float angle = -2.0f * M_PI / size;
        std::complex<float> wn(std::cos(angle), std::sin(angle));
        
        for (int i = 0; i < N; i += size) {
            std::complex<float> w(1.0f, 0.0f);
            
            for (int j = 0; j < halfSize; ++j) {
                std::complex<float> t = w * x[i + j + halfSize];
                std::complex<float> u = x[i + j];
                
                x[i + j] = u + t;
                x[i + j + halfSize] = u - t;
                
                w *= wn;
            }
        }
    }
    
    // Copy positive frequencies only
    for (int k = 0; k <= N / 2; ++k) {
        spectrum[k] = x[k];
    }
#endif
}

void FeatureExtractor::computePowerSpectrum(const std::vector<std::complex<float>>& spectrum,
                                             std::vector<float>& powerSpectrum) {
    /**
     * Power spectrum: P[k] = |X[k]|^2 / N
     */
    
    powerSpectrum.resize(spectrum.size());
    
    float normFactor = 1.0f / m_fftSize;
    
    for (size_t k = 0; k < spectrum.size(); ++k) {
        float re = spectrum[k].real();
        float im = spectrum[k].imag();
        powerSpectrum[k] = (re * re + im * im) * normFactor;
    }
}

void FeatureExtractor::applyMelFilterbank(const std::vector<float>& powerSpectrum,
                                           std::vector<float>& melEnergies) {
    /**
     * Áp dụng Mel filterbank
     * 
     * Mỗi Mel filter là một triangular filter trên frequency axis
     */
    
    melEnergies.resize(NUM_MEL_FILTERS, 0.0f);
    
    for (int m = 0; m < NUM_MEL_FILTERS; ++m) {
        float energy = 0.0f;
        
        // Chỉ duyệt trong range của filter
        int start = m_melFilterStart[m];
        int end = std::min(m_melFilterEnd[m], static_cast<int>(powerSpectrum.size()));
        
        for (int k = start; k < end; ++k) {
            energy += powerSpectrum[k] * m_melFilterbank[m][k];
        }
        
        melEnergies[m] = energy;
    }
}

void FeatureExtractor::computeDCT(const std::vector<float>& input,
                                   std::vector<float>& output,
                                   int numCoeffs) {
    /**
     * DCT-II (Discrete Cosine Transform Type 2)
     * 
     * Sử dụng pre-computed DCT matrix
     */
    
    output.resize(numCoeffs);
    
    for (int k = 0; k < numCoeffs; ++k) {
        float sum = 0.0f;
        
        for (int n = 0; n < static_cast<int>(input.size()) && n < NUM_MEL_FILTERS; ++n) {
            sum += input[n] * m_dctMatrix[k][n];
        }
        
        output[k] = sum;
    }
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

float FeatureExtractor::hzToMel(float freq) {
    /**
     * Chuyển đổi Hz sang Mel scale
     * 
     * Công thức: m = 2595 * log10(1 + f/700)
     */
    return 2595.0f * std::log10(1.0f + freq / 700.0f);
}

float FeatureExtractor::melToHz(float mel) {
    /**
     * Chuyển đổi Mel sang Hz
     * 
     * Công thức: f = 700 * (10^(m/2595) - 1)
     */
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

int FeatureExtractor::sign(float x) {
    /**
     * Sign function: sgn(x)
     * 
     * Trả về: 1 nếu x > 0, -1 nếu x < 0, 0 nếu x == 0
     */
    if (x > 0.0f) return 1;
    if (x < 0.0f) return -1;
    return 0;
}

// ============================================================================
// BATCH PROCESSING UTILITIES
// ============================================================================

void extractBatchFeatures(FeatureExtractor& extractor,
                          const std::vector<std::vector<float>>& allSamples,
                          std::vector<CycleFeatures>& allFeatures,
                          bool verbose) {
    allFeatures.clear();
    allFeatures.reserve(allSamples.size());
    
    size_t total = allSamples.size();
    size_t successCount = 0;
    
    for (size_t i = 0; i < total; ++i) {
        CycleFeatures features;
        
        if (extractor.extractFeatures(allSamples[i], features, false)) {
            allFeatures.push_back(std::move(features));
            successCount++;
        }
        
        if (verbose && (i + 1) % 100 == 0) {
            std::cout << "  Feature extraction: " << (i + 1) << "/" << total 
                      << " (" << successCount << " success)" << std::endl;
        }
    }
    
    if (verbose) {
        std::cout << "  Feature extraction complete: " << successCount 
                  << "/" << total << " cycles processed" << std::endl;
    }
}

void extractBatchFlatFeatures(FeatureExtractor& extractor,
                              const std::vector<std::vector<float>>& allSamples,
                              std::vector<std::vector<float>>& featureMatrix) {
    featureMatrix.clear();
    featureMatrix.reserve(allSamples.size());
    
    for (const auto& samples : allSamples) {
        std::vector<float> flat;
        
        if (extractor.extractFlatFeatures(samples, flat)) {
            featureMatrix.push_back(std::move(flat));
        }
    }
}

} // namespace respiratory
