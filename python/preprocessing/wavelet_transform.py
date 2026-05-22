import argparse
import os
import sys
from pathlib import Path

import numpy as np

import pywt

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import PROCESSED_AUDIO_DIR, SPECTROGRAMS_DIR



def generate_wavelet_spectrogram(signal: np.ndarray, scales: np.ndarray = np.arange(1, 128)) -> np.ndarray:

    """

    Converts a preprocessed 1D audio signal into a 224x224x3 RGB spectrogram

    using the Continuous Wavelet Transform (CWT) with a Morlet wavelet.



    Parameters:

        signal (np.ndarray): 1D array containing the preprocessed audio signal.

        scales (np.ndarray): Array of scales to use for CWT. Defaults to 1-127.



    Returns:

        np.ndarray: A 224x224x3 uint8 numpy array representing the RGB spectrogram.

    """

    # 1. Transform Method: Continuous Wavelet Transform (CWT) using Morlet wavelet ('morl')

    # Generate coefficients for the time-frequency distribution

    coefficients, _ = pywt.cwt(signal, scales, 'morl')



    # 2. Feature Extraction: Compute the absolute magnitude of the wavelet coefficients

    magnitude = np.abs(coefficients)



    # Apply log-scale transformation to better capture non-stationary patterns

    # np.log1p (log(1+x)) is used to handle safety against log(0)

    log_magnitude = np.log1p(magnitude)



    # Normalize to 8-bit integer range [0, 255]

    min_val = np.min(log_magnitude)

    max_val = np.max(log_magnitude)

   

    if max_val > min_val:

        normalized = 255.0 * (log_magnitude - min_val) / (max_val - min_val)

    else:

        normalized = np.zeros_like(log_magnitude)

       

    normalized_8bit = normalized.astype(np.uint8)



    # 3. CNN Formatting: Resize EXACTLY to 224x224 pixels using bicubic interpolation

    # cv2.resize expects target size as (width, height)

    # resized_spectrogram = cv2.resize(normalized_8bit, (224, 224), interpolation=cv2.INTER_CUBIC)



    # 4. Channel Requirement: Convert single-channel (grayscale) to 3-channel (RGB)

    # rgb_spectrogram = cv2.cvtColor(resized_spectrogram, cv2.COLOR_GRAY2RGB)
    colored_spectrogram = cv2.applyColorMap(normalized_8bit, cv2.COLORMAP_JET)

    resized_spectrogram = cv2.resize(colored_spectrogram, (224, 224), interpolation=cv2.INTER_CUBIC)

    return resized_spectrogram



def batch_generate_spectrograms(signals: list, scales: np.ndarray = np.arange(1, 128)) -> np.ndarray:

    """

    Efficiently processes a batch of 1D signals into 224x224x3 RGB spectrograms.



    Parameters:

        signals (list or np.ndarray): A batch (iterable) of 1D audio signals.

        scales (np.ndarray): Array of scales to use for CWT.



    Returns:

        np.ndarray: Array of shape (batch, 224, 224, 3) suitable for CNN input.

    """

    # List comprehension is highly optimized for iterating and collecting outputs

    # before combining them into a single batch numpy array.

    spectrograms = [generate_wavelet_spectrogram(sig, scales) for sig in signals]

    return np.stack(spectrograms)



if __name__ == "__main__":

    import librosa

    from tqdm import tqdm



    parser = argparse.ArgumentParser(description="Generate wavelet spectrograms from processed audio")

    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_AUDIO_DIR),
                        help="Directory containing processed audio class subfolders")

    parser.add_argument("--output_dir", type=str, default=str(SPECTROGRAMS_DIR),
                        help="Directory for generated spectrogram images")

    args = parser.parse_args()



    input_base_dir = args.data_dir

    output_base_dir = args.output_dir



    if not os.path.exists(input_base_dir):

        print(f"Input directory not found: {input_base_dir}")

        exit()



    subdirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]

    

    total_processed = 0

    total_failed = 0



    for category in subdirs:

        cat_dir = os.path.join(input_base_dir, category)

        out_cat_dir = os.path.join(output_base_dir, category)

        os.makedirs(out_cat_dir, exist_ok=True)

       

        files = [f for f in os.listdir(cat_dir) if f.endswith('.wav')]

        print(f"\nProcessing category: {category} ({len(files)} files)")

       

        for f in tqdm(files, desc=category, unit="file"):

            input_path = os.path.join(cat_dir, f)

            # Output will be a .png spectrogram image

            output_path = os.path.join(out_cat_dir, f.replace('.wav', '.png'))

           

            try:

                # Load the preprocessed audio (4kHz)

                signal, sr = librosa.load(input_path, sr=None)

               

                # If there's no audio data, skip

                if len(signal) == 0:

                    total_failed += 1

                    continue

                

                # Generate the RGB Morlet Wavelet Spectrogram

                spectrogram = generate_wavelet_spectrogram(signal)

               

                # Save out the resulting image

                cv2.imwrite(output_path, spectrogram)

                total_processed += 1

            except Exception as e:

                total_failed += 1



    print(f"\nBatch Spectrogram Generation Completed!")

    print(f"Successfully generated {total_processed} spectrograms.")

    print(f"Outputs are stored in: {output_base_dir}")

    if total_failed > 0:

        print(f"Failed to extract for {total_failed} files.")