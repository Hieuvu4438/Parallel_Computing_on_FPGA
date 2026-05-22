import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import COMBINED_AUDIO_DIR, COMBINED_LABELS, PROCESSED_AUDIO_DIR

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a bandpass filter to the data."""
    nyq = 0.5 * fs
    # Adjust highcut if it exceeds Nyquist frequency
    if highcut >= nyq:
        highcut = nyq - 1.0 
    
    # Safe check in case lowcut goes above highcut after adjustment
    if lowcut >= highcut:
        return data
        
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def normalize_signal(data):
    """Normalize amplitude to the range [-1, 1]."""
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data / max_val
    return data

def preprocess_file(input_path, output_path, target_sr=4000, lowcut=50, highcut=2500):
    """
    Apply preprocessing pipeline:
    1. Load audio with its original Sampling Rate
    2. Trim leading/trailing silences (noise reduction)
    3. Apply Band-pass Filter (50Hz - 2500Hz)
    4. Resample to 4kHz
    5. Amplitude Normalization to [-1, 1]
    """
    try:
        # 1. Load audio (original SR)
        audio, sr = librosa.load(input_path, sr=None)
        
        # Safety check: if audio is completely silent or empty
        if len(audio) == 0:
            return False
            
        # 2. Trim silence (Top 20dB) - additional preprocessing as per standard best practices
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # 3. Apply bandpass filter
        # It's better to filter at the original sampling rate for highcut=2500 to be valid.
        filtered_audio = bandpass_filter(audio, lowcut, highcut, sr)
        
        # 4. Resampling
        if sr != target_sr:
            resampled_audio = librosa.resample(filtered_audio, orig_sr=sr, target_sr=target_sr)
        else:
            resampled_audio = filtered_audio
            
        # 5. Normalization [-1, 1]
        normalized_audio = normalize_signal(resampled_audio)
        
        # Save processed file
        sf.write(output_path, normalized_audio, target_sr)
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Preprocess combined audio files")
    parser.add_argument("--data_dir", type=str, default=str(COMBINED_AUDIO_DIR),
                        help="Directory containing class-subfolder .wav files")
    parser.add_argument("--labels_csv", type=str, default=str(COMBINED_LABELS),
                        help="Combined labels.csv path")
    parser.add_argument("--output_dir", type=str, default=str(PROCESSED_AUDIO_DIR),
                        help="Directory for preprocessed .wav files")
    args = parser.parse_args()

    labels_file = args.labels_csv
    audio_dir = args.data_dir
    output_dir = args.output_dir

    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return
        
    # Read labels if necessary for further operations
    if os.path.exists(labels_file):
        df_labels = pd.read_csv(labels_file)
        print(f"Loaded labels for {len(df_labels)} audio records.")
        
    # Process files dynamically across existing subdirectories
    subdirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    
    success_count = 0
    total_files = 0
    
    for category in subdirs:
        cat_dir = os.path.join(audio_dir, category)
        out_cat_dir = os.path.join(output_dir, category)
        os.makedirs(out_cat_dir, exist_ok=True)
        
        files = [f for f in os.listdir(cat_dir) if f.endswith('.wav')]
        total_files += len(files)
        
        print(f"\nProcessing category: {category} ({len(files)} files)")
        for f in tqdm(files, desc=category, unit="file"):
            input_path = os.path.join(cat_dir, f)
            output_path = os.path.join(out_cat_dir, f)
            
            if preprocess_file(input_path, output_path):
                success_count += 1
                
    print(f"\nPreprocessing completed successfully.")
    print(f"Processed {success_count}/{total_files} files.")
    print(f"Outputs are stored in: {output_dir}")

if __name__ == "__main__":
    main()
