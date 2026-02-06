/**
 * @file WaveletTransform.cpp
 * @brief Implementation of Wavelet Transform for Spectrogram Generation
 * 
 * Triển khai biến đổi Wavelet liên tục (CWT) với Morlet wavelet
 * để tạo ảnh spectrogram đa phân giải cho CNN input.
 * 
 * @author Research Team
 * @date 2026
 */

#include "WaveletTransform.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <iostream>
#include <cassert>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace respiratory {

// ============================================================================
// CONSTANTS
// ============================================================================

constexpr float PI = 3.14159265358979323846f;
constexpr float EPSILON = 1e-10f;

// ============================================================================
// SPECTROGRAM IMPLEMENTATION
// ============================================================================

std::vector<float> Spectrogram::toNCHW() const {
    // Layout: [1, C, H, W] - Batch=1
    std::vector<float> nchw(data.size());
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int srcIdx = (h * width + w) * channels + c;
                int dstIdx = c * height * width + h * width + w;
                nchw[dstIdx] = data[srcIdx];
            }
        }
    }
    
    return nchw;
}

std::vector<float> Spectrogram::toNHWC() const {
    // Layout: [1, H, W, C] - already in this format
    return data;
}

// ============================================================================
// WAVELET TRANSFORM IMPLEMENTATION
// ============================================================================

WaveletTransform::WaveletTransform(const WaveletConfig& config)
    : m_config(config)
    , m_isInitialized(false)
{
    initializeScales();
}

WaveletTransform::~WaveletTransform() = default;

void WaveletTransform::initializeScales() {
    /**
     * Tính toán scales từ dải tần số mong muốn
     * 
     * Với Morlet wavelet:
     *   f = (ω₀ + √(2 + ω₀²)) / (4πs)
     * 
     * Suy ra:
     *   s = (ω₀ + √(2 + ω₀²)) / (4πf)
     */
    
    m_scales.clear();
    m_frequencies.clear();
    
    float omega0 = m_config.morletOmega0;
    float dt = 1.0f / m_config.sampleRate;
    
    // Tính scale_min và scale_max từ frequency range
    float scaleMin = frequencyToScale(m_config.maxFreq);
    float scaleMax = frequencyToScale(m_config.minFreq);
    
    // Tạo scales theo log-space (tốt hơn linear-space cho âm thanh)
    float logScaleMin = std::log2(scaleMin);
    float logScaleMax = std::log2(scaleMax);
    
    for (int i = 0; i < m_config.numScales; ++i) {
        float t = static_cast<float>(i) / (m_config.numScales - 1);
        float logScale = logScaleMin + t * (logScaleMax - logScaleMin);
        float scale = std::pow(2.0f, logScale);
        
        m_scales.push_back(scale);
        m_frequencies.push_back(scaleToFrequency(scale));
    }
    
    m_isInitialized = true;
}

float WaveletTransform::scaleToFrequency(float scale) const {
    /**
     * Pseudo-frequency của Morlet wavelet
     * f = (ω₀ + √(2 + ω₀²)) / (4π × scale × dt)
     * 
     * Xấp xỉ: f ≈ ω₀ / (2π × scale × dt) khi ω₀ >> 1
     */
    float omega0 = m_config.morletOmega0;
    float dt = 1.0f / m_config.sampleRate;
    
    // Công thức chính xác
    float numerator = omega0 + std::sqrt(2.0f + omega0 * omega0);
    return numerator / (4.0f * PI * scale * dt);
}

float WaveletTransform::frequencyToScale(float frequency) const {
    float omega0 = m_config.morletOmega0;
    float dt = 1.0f / m_config.sampleRate;
    
    float numerator = omega0 + std::sqrt(2.0f + omega0 * omega0);
    return numerator / (4.0f * PI * frequency * dt);
}

bool WaveletTransform::transform(const std::vector<float>& signal, 
                                  Spectrogram& spectrogram) {
    if (signal.empty()) {
        return false;
    }
    
    // Step 1: Compute CWT coefficients
    CWTCoefficients coeffs;
    if (!computeCWT(signal, coeffs)) {
        return false;
    }
    
    // Step 2: Convert to spectrogram
    if (!coefficientsToSpectrogram(coeffs, spectrogram)) {
        return false;
    }
    
    return true;
}

bool WaveletTransform::computeCWT(const std::vector<float>& signal, 
                                   CWTCoefficients& coeffs) {
    /**
     * Continuous Wavelet Transform
     * 
     * W(s, τ) = (1/√s) ∫ x(t) × ψ*((t-τ)/s) dt
     * 
     * Trong đó:
     * - s: scale
     * - τ: translation (time shift)
     * - ψ: mother wavelet (Morlet)
     */
    
    if (signal.empty() || !m_isInitialized) {
        return false;
    }
    
    int signalLength = static_cast<int>(signal.size());
    int numScales = static_cast<int>(m_scales.size());
    
    coeffs.numScales = numScales;
    coeffs.numTimePoints = signalLength;
    coeffs.scales = m_scales;
    coeffs.frequencies = m_frequencies;
    coeffs.coeffs.resize(numScales);
    
    // Parallel processing over scales
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int s = 0; s < numScales; ++s) {
        float scale = m_scales[s];
        
        // Compute wavelet at this scale
        // Wavelet length scales with the scale parameter
        int waveletLength = static_cast<int>(std::ceil(scale * 10.0f));
        waveletLength = std::min(waveletLength, signalLength);
        waveletLength = std::max(waveletLength, 8);
        
        // Make it odd for symmetric wavelet
        if (waveletLength % 2 == 0) waveletLength++;
        
        auto wavelet = computeMorletWavelet(scale, waveletLength);
        
        // Convolve signal with wavelet
        coeffs.coeffs[s] = convolveWithWavelet(signal, wavelet);
    }
    
    return true;
}

std::vector<std::complex<float>> WaveletTransform::computeMorletWavelet(
    float scale, int length) {
    /**
     * Morlet (Gabor) Wavelet:
     * 
     * ψ(η) = π^(-1/4) × exp(-η²/2) × (exp(iω₀η) - exp(-ω₀²/2))
     * 
     * Simplified (khi ω₀ ≥ 5):
     * ψ(η) = π^(-1/4) × exp(-η²/2) × exp(iω₀η)
     * 
     * Với η = (t - t₀) / s
     */
    
    std::vector<std::complex<float>> wavelet(length);
    
    float omega0 = m_config.morletOmega0;
    float normFactor = std::pow(PI, -0.25f) / std::sqrt(scale);
    
    int center = length / 2;
    
    for (int i = 0; i < length; ++i) {
        // η = (i - center) / scale
        float eta = static_cast<float>(i - center) / scale;
        
        // Gaussian envelope
        float gaussian = std::exp(-0.5f * eta * eta);
        
        // Complex sinusoid
        float realPart = std::cos(omega0 * eta);
        float imagPart = std::sin(omega0 * eta);
        
        // Wavelet value
        wavelet[i] = normFactor * gaussian * std::complex<float>(realPart, imagPart);
    }
    
    return wavelet;
}

std::vector<float> WaveletTransform::computeMexicanHat(float scale, int length) {
    /**
     * Mexican Hat (Ricker) Wavelet:
     * 
     * ψ(t) = (2/(√3σ×π^(1/4))) × (1 - (t/σ)²) × exp(-t²/(2σ²))
     * 
     * σ = scale
     */
    
    std::vector<float> wavelet(length);
    
    float sigma = scale;
    float normFactor = 2.0f / (std::sqrt(3.0f * sigma) * std::pow(PI, 0.25f));
    
    int center = length / 2;
    
    for (int i = 0; i < length; ++i) {
        float t = static_cast<float>(i - center);
        float tNorm = t / sigma;
        float tNorm2 = tNorm * tNorm;
        
        wavelet[i] = normFactor * (1.0f - tNorm2) * std::exp(-0.5f * tNorm2);
    }
    
    return wavelet;
}

std::vector<std::complex<float>> WaveletTransform::convolveWithWavelet(
    const std::vector<float>& signal,
    const std::vector<std::complex<float>>& wavelet) {
    /**
     * Direct convolution (đơn giản, phù hợp với HLS)
     * 
     * Có thể thay bằng FFT convolution cho signals dài
     */
    
    int signalLen = static_cast<int>(signal.size());
    int waveletLen = static_cast<int>(wavelet.size());
    
    std::vector<std::complex<float>> result(signalLen);
    
    int halfWavelet = waveletLen / 2;
    
    for (int i = 0; i < signalLen; ++i) {
        std::complex<float> sum(0.0f, 0.0f);
        
        for (int j = 0; j < waveletLen; ++j) {
            int signalIdx = i - halfWavelet + j;
            
            // Zero-padding at boundaries
            if (signalIdx >= 0 && signalIdx < signalLen) {
                // Conjugate of wavelet for convolution
                sum += signal[signalIdx] * std::conj(wavelet[j]);
            }
        }
        
        result[i] = sum;
    }
    
    return result;
}

bool WaveletTransform::coefficientsToSpectrogram(const CWTCoefficients& coeffs,
                                                   Spectrogram& spectrogram) {
    if (coeffs.coeffs.empty()) {
        return false;
    }
    
    int numScales = coeffs.numScales;
    int numTimePoints = coeffs.numTimePoints;
    
    // Step 1: Compute scalogram (magnitude/power)
    std::vector<std::vector<float>> scalogram(numScales, 
                                               std::vector<float>(numTimePoints));
    
    for (int s = 0; s < numScales; ++s) {
        for (int t = 0; t < numTimePoints; ++t) {
            // Use power (magnitude squared)
            scalogram[s][t] = coeffs.getPower(s, t);
        }
    }
    
    // Step 2: Resize to output dimensions
    std::vector<std::vector<float>> resized;
    resizeScalogram(scalogram, resized, m_config.outputWidth, m_config.outputHeight);
    
    // Step 3: Convert to spectrogram structure
    spectrogram.allocate(m_config.outputWidth, m_config.outputHeight, 
                         m_config.numChannels);
    spectrogram.frequencies = coeffs.frequencies;
    
    // Compute time axis
    float dt = 1.0f / m_config.sampleRate;
    spectrogram.times.resize(numTimePoints);
    for (int t = 0; t < numTimePoints; ++t) {
        spectrogram.times[t] = t * dt;
    }
    
    // Copy data (inverted: low freq at bottom for visual consistency)
    for (int y = 0; y < m_config.outputHeight; ++y) {
        for (int x = 0; x < m_config.outputWidth; ++x) {
            // Invert y-axis: low frequencies at bottom
            int srcY = m_config.outputHeight - 1 - y;
            spectrogram.set(x, y, resized[srcY][x]);
        }
    }
    
    // Step 4: Normalize
    normalizeSpectrogram(spectrogram);
    
    return true;
}

void WaveletTransform::resizeScalogram(const std::vector<std::vector<float>>& input,
                                        std::vector<std::vector<float>>& output,
                                        int newWidth, int newHeight) {
    /**
     * Bilinear interpolation resize
     */
    
    int oldHeight = static_cast<int>(input.size());
    int oldWidth = static_cast<int>(input[0].size());
    
    output.resize(newHeight, std::vector<float>(newWidth));
    
    float xScale = static_cast<float>(oldWidth - 1) / (newWidth - 1);
    float yScale = static_cast<float>(oldHeight - 1) / (newHeight - 1);
    
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            float srcX = x * xScale;
            float srcY = y * yScale;
            
            output[y][x] = bilinearInterpolate(input, srcX, srcY);
        }
    }
}

float WaveletTransform::bilinearInterpolate(
    const std::vector<std::vector<float>>& data,
    float x, float y) {
    
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    int maxY = static_cast<int>(data.size()) - 1;
    int maxX = static_cast<int>(data[0].size()) - 1;
    
    x0 = std::clamp(x0, 0, maxX);
    x1 = std::clamp(x1, 0, maxX);
    y0 = std::clamp(y0, 0, maxY);
    y1 = std::clamp(y1, 0, maxY);
    
    float xFrac = x - std::floor(x);
    float yFrac = y - std::floor(y);
    
    float v00 = data[y0][x0];
    float v01 = data[y0][x1];
    float v10 = data[y1][x0];
    float v11 = data[y1][x1];
    
    float v0 = v00 * (1.0f - xFrac) + v01 * xFrac;
    float v1 = v10 * (1.0f - xFrac) + v11 * xFrac;
    
    return v0 * (1.0f - yFrac) + v1 * yFrac;
}

void WaveletTransform::normalizeSpectrogram(Spectrogram& spectrogram) {
    switch (m_config.normType) {
        case NormalizationType::LOG_SCALE:
            applyLogTransform(spectrogram);
            break;
        case NormalizationType::POWER_TO_DB:
            powerToDb(spectrogram);
            break;
        case NormalizationType::Z_SCORE: {
            // Z-score normalization
            float sum = 0.0f, sumSq = 0.0f;
            for (float v : spectrogram.data) {
                sum += v;
                sumSq += v * v;
            }
            float mean = sum / spectrogram.data.size();
            float variance = sumSq / spectrogram.data.size() - mean * mean;
            float stddev = std::sqrt(variance + EPSILON);
            
            for (float& v : spectrogram.data) {
                v = (v - mean) / stddev;
            }
            break;
        }
        case NormalizationType::MIN_MAX:
        default: {
            // Min-max normalization to [0, 1]
            float minVal = *std::min_element(spectrogram.data.begin(), 
                                              spectrogram.data.end());
            float maxVal = *std::max_element(spectrogram.data.begin(), 
                                              spectrogram.data.end());
            
            spectrogram.minValue = minVal;
            spectrogram.maxValue = maxVal;
            
            float range = maxVal - minVal;
            if (range > EPSILON) {
                for (float& v : spectrogram.data) {
                    v = (v - minVal) / range;
                }
            }
            break;
        }
    }
}

void WaveletTransform::applyLogTransform(Spectrogram& spectrogram) {
    /**
     * Log transform: log(1 + x)
     * Sau đó normalize về [0, 1]
     */
    
    // Apply log
    for (float& v : spectrogram.data) {
        v = std::log1p(std::abs(v));
    }
    
    // Normalize
    float minVal = *std::min_element(spectrogram.data.begin(), 
                                      spectrogram.data.end());
    float maxVal = *std::max_element(spectrogram.data.begin(), 
                                      spectrogram.data.end());
    
    spectrogram.minValue = minVal;
    spectrogram.maxValue = maxVal;
    
    float range = maxVal - minVal;
    if (range > EPSILON) {
        for (float& v : spectrogram.data) {
            v = (v - minVal) / range;
        }
    }
}

void WaveletTransform::powerToDb(Spectrogram& spectrogram, float refPower) {
    /**
     * Convert power to dB: 10 × log10(power / refPower)
     * Clamp to [-80, 0] dB range, then normalize
     */
    
    constexpr float minDb = -80.0f;
    constexpr float maxDb = 0.0f;
    
    for (float& v : spectrogram.data) {
        float powerDb = 10.0f * std::log10(std::max(v, EPSILON) / refPower);
        powerDb = std::clamp(powerDb, minDb, maxDb);
        
        // Normalize to [0, 1]
        v = (powerDb - minDb) / (maxDb - minDb);
    }
    
    spectrogram.minValue = minDb;
    spectrogram.maxValue = maxDb;
}

void WaveletTransform::setConfig(const WaveletConfig& config) {
    m_config = config;
    initializeScales();
}

void WaveletTransform::setWaveletType(WaveletType type) {
    m_config.waveletType = type;
}

void WaveletTransform::setNormalizationType(NormalizationType type) {
    m_config.normType = type;
}

void WaveletTransform::setOutputSize(int width, int height) {
    m_config.outputWidth = width;
    m_config.outputHeight = height;
}

void WaveletTransform::setFrequencyRange(float minFreq, float maxFreq) {
    m_config.minFreq = minFreq;
    m_config.maxFreq = maxFreq;
    initializeScales();  // Re-compute scales
}

std::vector<Spectrogram> WaveletTransform::transformBatch(
    const std::vector<std::vector<float>>& signals) {
    
    std::vector<Spectrogram> results(signals.size());
    
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < signals.size(); ++i) {
        transform(signals[i], results[i]);
    }
    
    return results;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

Spectrogram createSpectrogram(const std::vector<float>& signal, int sampleRate) {
    WaveletConfig config;
    config.sampleRate = sampleRate;
    
    WaveletTransform transform(config);
    
    Spectrogram spec;
    transform.transform(signal, spec);
    
    return spec;
}

float compareSpectrograms(const Spectrogram& a, const Spectrogram& b) {
    if (a.size() != b.size()) {
        return 0.0f;
    }
    
    // Cosine similarity
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dotProduct += a.data[i] * b.data[i];
        normA += a.data[i] * a.data[i];
        normB += b.data[i] * b.data[i];
    }
    
    return dotProduct / (std::sqrt(normA) * std::sqrt(normB) + EPSILON);
}

bool saveSpectrogram(const Spectrogram& spec, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header
    file.write(reinterpret_cast<const char*>(&spec.width), sizeof(int));
    file.write(reinterpret_cast<const char*>(&spec.height), sizeof(int));
    file.write(reinterpret_cast<const char*>(&spec.channels), sizeof(int));
    
    // Write data
    file.write(reinterpret_cast<const char*>(spec.data.data()), 
               spec.data.size() * sizeof(float));
    
    return true;
}

bool loadSpectrogram(Spectrogram& spec, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Read header
    file.read(reinterpret_cast<char*>(&spec.width), sizeof(int));
    file.read(reinterpret_cast<char*>(&spec.height), sizeof(int));
    file.read(reinterpret_cast<char*>(&spec.channels), sizeof(int));
    
    // Read data
    spec.data.resize(spec.width * spec.height * spec.channels);
    file.read(reinterpret_cast<char*>(spec.data.data()), 
              spec.data.size() * sizeof(float));
    
    return true;
}

bool exportSpectrogramToCSV(const Spectrogram& spec, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    for (int y = 0; y < spec.height; ++y) {
        for (int x = 0; x < spec.width; ++x) {
            file << spec.at(x, y);
            if (x < spec.width - 1) file << ",";
        }
        file << "\n";
    }
    
    return true;
}

// ============================================================================
// FFT HELPERS (Simple implementation - can be replaced with FFTW3)
// ============================================================================

int WaveletTransform::nextPowerOf2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

std::vector<std::complex<float>> WaveletTransform::zeroPad(
    const std::vector<std::complex<float>>& input, int newLength) {
    
    std::vector<std::complex<float>> padded(newLength, std::complex<float>(0.0f, 0.0f));
    std::copy(input.begin(), input.end(), padded.begin());
    return padded;
}

std::vector<std::complex<float>> WaveletTransform::dft(
    const std::vector<std::complex<float>>& input) {
    /**
     * Simple DFT implementation (O(n²))
     * For production, replace with FFTW3
     */
    
    int N = static_cast<int>(input.size());
    std::vector<std::complex<float>> output(N);
    
    for (int k = 0; k < N; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * PI * k * n / N;
            sum += input[n] * std::complex<float>(std::cos(angle), std::sin(angle));
        }
        output[k] = sum;
    }
    
    return output;
}

std::vector<std::complex<float>> WaveletTransform::idft(
    const std::vector<std::complex<float>>& input) {
    /**
     * Simple IDFT implementation
     */
    
    int N = static_cast<int>(input.size());
    std::vector<std::complex<float>> output(N);
    
    for (int n = 0; n < N; ++n) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int k = 0; k < N; ++k) {
            float angle = 2.0f * PI * k * n / N;
            sum += input[k] * std::complex<float>(std::cos(angle), std::sin(angle));
        }
        output[n] = sum / static_cast<float>(N);
    }
    
    return output;
}

std::vector<std::complex<float>> WaveletTransform::fftConvolve(
    const std::vector<float>& signal,
    const std::vector<std::complex<float>>& wavelet) {
    /**
     * FFT-based convolution (faster for long signals)
     * 
     * conv(x, h) = IFFT(FFT(x) × FFT(h))
     */
    
    int convLen = static_cast<int>(signal.size() + wavelet.size() - 1);
    int fftLen = nextPowerOf2(convLen);
    
    // Zero-pad signal
    std::vector<std::complex<float>> signalPadded(fftLen, std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < signal.size(); ++i) {
        signalPadded[i] = std::complex<float>(signal[i], 0.0f);
    }
    
    // Zero-pad wavelet
    std::vector<std::complex<float>> waveletPadded = zeroPad(wavelet, fftLen);
    
    // FFT
    auto signalFFT = dft(signalPadded);
    auto waveletFFT = dft(waveletPadded);
    
    // Multiply
    std::vector<std::complex<float>> productFFT(fftLen);
    for (int i = 0; i < fftLen; ++i) {
        productFFT[i] = signalFFT[i] * std::conj(waveletFFT[i]);
    }
    
    // IFFT
    auto result = idft(productFFT);
    
    // Trim to signal length
    result.resize(signal.size());
    
    return result;
}

} // namespace respiratory
