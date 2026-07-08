"""
Extract intermediate features from CLAP teacher ensemble for feature-level distillation.

The CLAP teacher produces 1024-dim features (512 text + 512 audio).
We extract these features and store them alongside teacher logits.

Usage:
    python util/teacher_features.py --data_folder ./data/ --n_teachers 5

Output:
    teacher_features/teacher_features.training.pt  [N_samples, n_teachers, 1024]
    teacher_features/teacher_features.test.pt       [N_samples, n_teachers, 1024]
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import ClapProcessor, ClapModel


def extract_teacher_features(args):
    """Extract features from CLAP teacher ensemble."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CLAP model
    print("Loading CLAP model...")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", sampling_rate=48000)
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    model.eval()

    # Load existing training/test data (already preprocessed)
    for split in ['training', 'test']:
        pt_path = f'./data/{split}.pt'
        if not os.path.exists(pt_path):
            print(f"Warning: {pt_path} not found, skipping")
            continue

        print(f"\nProcessing {split} set...")
        data = torch.load(pt_path, weights_only=False)
        print(f"  Loaded {len(data)} samples")

        features_list = []
        with torch.no_grad():
            for i, sample in enumerate(data):
                audio_input = sample[0].unsqueeze(0).to(device)  # [1, 1, audio_len] or similar

                # Get CLAP features
                clap_output = model(input_features=audio_input)
                # text_embeds + audio_embeds = 512 + 512 = 1024
                feature = torch.cat([clap_output.text_embeds, clap_output.audio_embeds], dim=-1)
                features_list.append(feature.cpu())

                if (i + 1) % 100 == 0:
                    print(f"  Processed {i+1}/{len(data)} samples")

        # Stack all features
        all_features = torch.cat(features_list, dim=0)  # [N, 1024]
        print(f"  Feature shape: {all_features.shape}")

        # Save
        os.makedirs('teacher_features', exist_ok=True)
        save_path = f'teacher_features/teacher_features.{split}.pt'
        torch.save(all_features, save_path)
        print(f"  Saved to {save_path}")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./data/')
    args = parser.parse_args()
    extract_teacher_features(args)
