"""
Extract logits from a trained CLAP model checkpoint.
Saves logits for both training and test sets.

Usage:
    python scripts/extract_logits.py \
        --checkpoint save/icbhi_laion_clap-htsat-unfused_ce_all_BTS_larger_clap_1/best.pth \
        --model laion/larger_clap_general \
        --model_type ClapModel \
        --output_dir teacher_logits_larger_clap \
        --num_workers 4

Output:
    <output_dir>/teacher_logits.training.pt   [N_train, n_cls]
    <output_dir>/teacher_logits.test.pt       [N_test, n_cls]
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_custom import _is_clap, _patched_set_loader
from models.clap_larger import PretrainedCLAP as LargerPretrainedCLAP
from models.clap import PretrainedCLAP as OriginalPretrainedCLAP
from transformers import ClapProcessor


def extract_logits(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    if args.model == 'laion/larger_clap_general':
        if args.model_type == 'ClapModel':
            model = LargerPretrainedCLAP(args.model, 512)
        else:
            from models.clap_larger import PretrainedCLAPWithProjection
            model = PretrainedCLAPWithProjection(args.model, 512)
    else:
        if args.model_type == 'ClapModel':
            model = OriginalPretrainedCLAP(args.model, 512)
        else:
            from models.clap import PretrainedCLAPWithProjection
            model = PretrainedCLAPWithProjection(args.model, 512)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()

    # Build classifier
    if args.model_type == 'ClapModel':
        if args.clap_final == 'concat':
            classifier = nn.Linear(1024, args.n_cls)
        else:
            classifier = nn.Linear(512, args.n_cls)
    else:
        classifier = nn.Linear(512, args.n_cls)

    if 'classifier' in checkpoint:
        classifier.load_state_dict(checkpoint['classifier'])
    classifier = classifier.to(device)
    classifier.eval()

    # Process each split
    for split in ['train', 'test']:
        cache_path = f'./data/training.pt' if split == 'train' else './data/test.pt'
        if not os.path.exists(cache_path):
            print(f"Warning: {cache_path} not found, skipping {split}")
            continue

        data = torch.load(cache_path, weights_only=False)
        print(f"\nProcessing {split} set: {len(data)} samples")

        logits_list = []
        with torch.no_grad():
            for i, sample in enumerate(data):
                audio = sample[0].unsqueeze(0).to(device)  # [1, 1, 1001, 64]

                if args.model_type == 'ClapModel':
                    # For ClapModel, we need text inputs too
                    # Use dummy text since we only need the forward pass
                    # Actually, the cached data doesn't have text — we need the dataset
                    # Let's use audio-only mode for extraction
                    from models.clap_larger import PretrainedCLAPWithProjection
                    audio_model = PretrainedCLAPWithProjection(args.model, 512)
                    audio_model.load_state_dict(
                        {k.replace('audio_features.', ''): v
                         for k, v in checkpoint['model'].items()
                         if k.startswith('audio_features.')},
                        strict=False,
                    )
                    audio_model = audio_model.to(device)
                    audio_model.eval()
                    features = audio_model(audio)
                    output = classifier(features)
                else:
                    features = model(audio)
                    output = classifier(features)

                logits_list.append(output.cpu())

                if (i + 1) % 200 == 0:
                    print(f"  Processed {i+1}/{len(data)}")

        all_logits = torch.cat(logits_list, dim=0)
        print(f"  Logits shape: {all_logits.shape}")

        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f'teacher_logits.{split if split == "train" else "test"}.pt')
        torch.save(all_logits.numpy(), save_path)
        print(f"  Saved to {save_path}")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model', type=str, default='laion/larger_clap_general')
    parser.add_argument('--model_type', type=str, default='ClapModel')
    parser.add_argument('--clap_final', type=str, default='concat')
    parser.add_argument('--n_cls', type=int, default=4)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    extract_logits(args)
