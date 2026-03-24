#!/usr/bin/env python3
"""
combine_dataset.py
==================
Gộp 2 dataset âm thanh hô hấp (ICBHI + samples_02) thành 1 dataset thống nhất
với 3 nhãn: Healthy, COPD, Non-COPD.

Output:
  - data/combined/audio/          : Tất cả file .wav (copy hoặc symlink)
  - data/combined/labels.csv      : CSV chứa filename, original_label, label, source
  - In summary thống kê ra terminal

Usage:
  python3 python/combine_dataset.py [--copy | --symlink] [--output-dir <path>]
"""

import os       
import sys
import csv
import shutil
import argparse
import glob
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset 1: ICBHI 2017
ICBHI_AUDIO_DIR = PROJECT_ROOT / "data" / "samples" / "ICBHI_final_database"
ICBHI_LABELS_FILE = PROJECT_ROOT / "data" / "samples" / "labels.txt"

# Dataset 2: samples_02
DS2_AUDIO_DIR = PROJECT_ROOT / "data" / "samples_02" / "Audio Files"
DS2_ANNOTATION = PROJECT_ROOT / "data" / "samples_02" / "Data annotation.xlsx"

# Output mặc định
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "combined"


# ---------------------------------------------------------------------------
# Bảng ánh xạ nhãn -> 3 classes: Healthy, COPD, Non-COPD
# ---------------------------------------------------------------------------
def map_to_3class(original_label: str) -> str:
    """
    Ánh xạ nhãn gốc từ cả 2 dataset sang 3 lớp: Healthy, COPD, Non-COPD.

    Quy tắc:
    - "Healthy" / "N" (Normal)                        -> Healthy
    - "COPD" / "copd"                                 -> COPD
    - Tất cả bệnh khác (URTI, Asthma, Bronchiectasis,
      Bronchiolitis, LRTI, Pneumonia, BRON, Heart Failure,
      Lung Fibrosis, Plueral Effusion, combo...)      -> Non-COPD
    """
    label_lower = original_label.strip().lower()

    # Healthy
    if label_lower in ("healthy", "n", "normal"):
        return "Healthy"

    # COPD (pure COPD only, not combo like "Heart Failure + COPD")
    if label_lower == "copd":
        return "COPD"

    # Tất cả các bệnh khác -> Non-COPD
    return "Non-COPD"


# ---------------------------------------------------------------------------
# Parse Dataset 1: ICBHI 2017
# ---------------------------------------------------------------------------
def parse_icbhi_dataset() -> list[dict]:
    """
    Đọc labels.txt và quét thư mục audio ICBHI.

    labels.txt format:
        patient_id<TAB>Diagnosis
        101\tURTI
        102\tHealthy

    Tên file audio: {patient_id}_{...}.wav
    Mỗi patient_id có nhiều file audio (ghi ở nhiều vị trí, nhiều lần).

    Returns:
        List of dicts: {filename, filepath, patient_id, original_label, label, source}
    """
    # 1. Đọc mapping patient_id -> diagnosis
    patient_diagnosis: dict[str, str] = {}
    if not ICBHI_LABELS_FILE.exists():
        print(f"[ERROR] Không tìm thấy file labels: {ICBHI_LABELS_FILE}")
        sys.exit(1)

    with open(ICBHI_LABELS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pid = parts[0].strip()
                diag = parts[1].strip()
                patient_diagnosis[pid] = diag

    print(f"[ICBHI] Đã đọc {len(patient_diagnosis)} bệnh nhân từ labels.txt")

    # 2. Quét tất cả .wav files
    records = []
    wav_files = sorted(ICBHI_AUDIO_DIR.glob("*.wav"))

    if not wav_files:
        print(f"[WARNING] Không tìm thấy file .wav trong {ICBHI_AUDIO_DIR}")
        return records

    skipped = 0
    for wav_path in wav_files:
        filename = wav_path.name
        # Tách patient_id từ tên file: "101_1b1_Al_sc_Meditron.wav" -> "101"
        patient_id = filename.split("_")[0]

        if patient_id not in patient_diagnosis:
            skipped += 1
            continue

        original_label = patient_diagnosis[patient_id]
        label = map_to_3class(original_label)

        records.append({
            "filename": filename,
            "filepath": str(wav_path),
            "patient_id": f"ICBHI_{patient_id}",
            "original_label": original_label,
            "label": label,
            "source": "ICBHI",
        })

    print(f"[ICBHI] Tìm thấy {len(wav_files)} file .wav, "
          f"matched {len(records)}, skipped {skipped}")

    return records


# ---------------------------------------------------------------------------
# Parse Dataset 2: samples_02
# ---------------------------------------------------------------------------
def parse_ds2_dataset() -> list[dict]:
    """
    Đọc nhãn từ file Excel annotation và gán cho các file audio.

    File audio có tên dạng:
        BP{id}_{Diagnosis},{sound_type},{location},{age},{gender}.wav
        DP{id}_{Diagnosis},...  (duplicate prefix)
        EP{id}_{Diagnosis},...  (triplicate prefix)

    Mỗi sample id (1-112) tương ứng 1 row trong Excel,
    nhưng có 3 bản ghi âm (BP, DP, EP).

    Strategy: Parse nhãn trực tiếp từ tên file (đáng tin cậy hơn vì
    tên file khớp chính xác với audio), cross-validate với Excel.

    Returns:
        List of dicts: {filename, filepath, patient_id, original_label, label, source}
    """
    records = []

    # 1. Đọc annotation từ Excel để cross-validate
    excel_diagnoses: dict[int, str] = {}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(DS2_ANNOTATION), read_only=True)
        ws = wb.active
        for idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=1):
            if row[4] is not None:
                excel_diagnoses[idx] = str(row[4]).strip()
        wb.close()
        print(f"[DS2] Đã đọc {len(excel_diagnoses)} mẫu từ Excel annotation")
    except ImportError:
        print("[WARNING] openpyxl chưa cài. Sẽ parse nhãn từ tên file.")
    except Exception as e:
        print(f"[WARNING] Không đọc được Excel: {e}. Sẽ parse nhãn từ tên file.")

    # 2. Quét .wav files và parse nhãn từ tên file
    if not DS2_AUDIO_DIR.exists():
        print(f"[ERROR] Không tìm thấy thư mục: {DS2_AUDIO_DIR}")
        return records

    wav_files = sorted(DS2_AUDIO_DIR.glob("*.wav"))

    if not wav_files:
        print(f"[WARNING] Không tìm thấy file .wav trong {DS2_AUDIO_DIR}")
        return records

    for wav_path in wav_files:
        filename = wav_path.name
        # Parse: "BP108_COPD,E W,P R L ,63,M.wav"
        #   prefix_id = "BP108"
        #   rest = "COPD,E W,P R L ,63,M"
        name_no_ext = wav_path.stem  # bỏ .wav

        # Tách prefix_id và phần thông tin
        underscore_idx = name_no_ext.find("_")
        if underscore_idx == -1:
            continue

        prefix_id = name_no_ext[:underscore_idx]     # e.g. "BP108"
        info_part = name_no_ext[underscore_idx + 1:]  # e.g. "COPD,E W,P R L ,63,M"

        # Tách diagnosis (phần đầu tiên trước dấu phẩy)
        comma_idx = info_part.find(",")
        if comma_idx == -1:
            continue

        diagnosis_raw = info_part[:comma_idx].strip()

        # Tách prefix và sample number
        prefix = ""
        sample_num_str = ""
        for i, ch in enumerate(prefix_id):
            if ch.isdigit():
                prefix = prefix_id[:i]
                sample_num_str = prefix_id[i:]
                break

        if not sample_num_str:
            continue

        # Ánh xạ nhãn
        original_label = _normalize_ds2_diagnosis(diagnosis_raw)
        label = map_to_3class(original_label)

        records.append({
            "filename": filename,
            "filepath": str(wav_path),
            "patient_id": f"DS2_{sample_num_str}_{prefix}",
            "original_label": original_label,
            "label": label,
            "source": "DS2",
        })

    print(f"[DS2] Tìm thấy {len(wav_files)} file .wav, matched {len(records)}")

    return records


def _normalize_ds2_diagnosis(raw: str) -> str:
    """
    Chuẩn hóa nhãn bệnh từ dataset 2 cho nhất quán.

    Ví dụ:
        "N"                          -> "Healthy"
        "COPD"                       -> "COPD"
        "copd"                       -> "COPD"
        "Asthma"                     -> "Asthma"
        "asthma"                     -> "Asthma"
        "Heart Failure"              -> "Heart Failure"
        "heart failure"              -> "Heart Failure"
        "Heart Failure + COPD"       -> "Heart Failure + COPD"
        "Heart Failure + Lung Fibrosis" -> "Heart Failure + Lung Fibrosis"
        "Plueral Effusion"           -> "Pleural Effusion"
        "Lung Fibrosis"              -> "Lung Fibrosis"
        "BRON"                       -> "Bronchitis"
        "pneumonia"                  -> "Pneumonia"
        "Asthma and lung fibrosis"   -> "Asthma and Lung Fibrosis"
    """
    label = raw.strip()
    label_lower = label.lower()

    # Direct mappings
    norm_map = {
        "n": "Healthy",
        "copd": "COPD",
        "asthma": "Asthma",
        "heart failure": "Heart Failure",
        "pneumonia": "Pneumonia",
        "bron": "Bronchitis",
        "lung fibrosis": "Pulmonary Fibrosis",
        "plueral effusion": "Pleural Effusion",
        "pleural effusion": "Pleural Effusion",
    }

    if label_lower in norm_map:
        return norm_map[label_lower]

    # Combo cases
    combo_map = {
        "heart failure + copd": "Heart Failure + COPD",
        "heart failure + lung fibrosis": "Heart Failure + Pulmonary Fibrosis",
        "asthma and lung fibrosis": "Asthma and Pulmonary Fibrosis",
    }

    if label_lower in combo_map:
        return combo_map[label_lower]

    # Fallback: capitalize
    return label.title()


# ---------------------------------------------------------------------------
# Combine & Export
# ---------------------------------------------------------------------------
def combine_datasets(
    output_dir: Path,
    mode: str = "symlink",
) -> None:
    """
    Gộp 2 dataset, tạo output directory với cấu trúc:

        output_dir/
        ├── audio/
        │   ├── Healthy/
        │   ├── COPD/
        │   └── Non-COPD/
        ├── labels.csv
        └── summary.txt
    """
    print("=" * 70)
    print("  COMBINE RESPIRATORY SOUND DATASETS")
    print("  3-Class: Healthy | COPD | Non-COPD")
    print("=" * 70)
    print()

    # 1. Parse cả 2 datasets
    records_icbhi = parse_icbhi_dataset()
    print()
    records_ds2 = parse_ds2_dataset()
    print()

    all_records = records_icbhi + records_ds2

    if not all_records:
        print("[ERROR] Không tìm thấy bất kỳ mẫu nào!")
        sys.exit(1)

    # 2. Tạo output directories
    audio_dir = output_dir / "audio"
    for label_name in ["Healthy", "COPD", "Non-COPD"]:
        (audio_dir / label_name).mkdir(parents=True, exist_ok=True)

    # 3. Copy/Symlink audio files & ghi CSV
    csv_path = output_dir / "labels.csv"
    duplicate_counter: dict[str, int] = {}

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "filename", "original_label", "label", "source", "patient_id"
        ])
        writer.writeheader()

        for rec in all_records:
            src_path = Path(rec["filepath"])
            if not src_path.exists():
                continue

            # Xử lý trùng tên file bằng cách thêm prefix source
            dest_filename = f"{rec['source']}_{rec['filename']}"

            # Kiểm tra trùng lặp
            if dest_filename in duplicate_counter:
                duplicate_counter[dest_filename] += 1
                stem = Path(dest_filename).stem
                ext = Path(dest_filename).suffix
                dest_filename = f"{stem}_dup{duplicate_counter[dest_filename]}{ext}"
            else:
                duplicate_counter[dest_filename] = 0

            dest_path = audio_dir / rec["label"] / dest_filename

            # Copy hoặc symlink
            if not dest_path.exists():
                if mode == "copy":
                    shutil.copy2(str(src_path), str(dest_path))
                elif mode == "symlink":
                    os.symlink(str(src_path), str(dest_path))
                else:  # "skip" - chỉ tạo CSV, không copy audio
                    pass

            # Ghi vào CSV
            writer.writerow({
                "filename": dest_filename,
                "original_label": rec["original_label"],
                "label": rec["label"],
                "source": rec["source"],
                "patient_id": rec["patient_id"],
            })

    # 4. Thống kê
    print_summary(all_records, output_dir)


def print_summary(records: list[dict], output_dir: Path) -> None:
    """In bảng thống kê chi tiết."""
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Tổng quan theo source
    source_counter = Counter(r["source"] for r in records)
    print(f"\n{'Source':<15} {'Count':>8}")
    print("-" * 25)
    for src, cnt in sorted(source_counter.items()):
        print(f"  {src:<13} {cnt:>8}")
    print(f"  {'TOTAL':<13} {len(records):>8}")

    # Thống kê theo 3-class label
    label_counter = Counter(r["label"] for r in records)
    print(f"\n{'3-Class Label':<15} {'Count':>8} {'Percent':>10}")
    print("-" * 35)
    for lbl in ["Healthy", "COPD", "Non-COPD"]:
        cnt = label_counter.get(lbl, 0)
        pct = cnt / len(records) * 100
        print(f"  {lbl:<13} {cnt:>8} {pct:>9.1f}%")
    print(f"  {'TOTAL':<13} {len(records):>8} {'100.0%':>10}")

    # Thống kê theo original label
    orig_counter = Counter(r["original_label"] for r in records)
    print(f"\n{'Original Label':<30} {'Count':>8} {'→ 3-Class':>15}")
    print("-" * 55)
    for lbl, cnt in sorted(orig_counter.items(), key=lambda x: -x[1]):
        mapped = map_to_3class(lbl)
        print(f"  {lbl:<28} {cnt:>8}   → {mapped}")

    # Thống kê chéo: source x label
    print(f"\n{'Source × Label':<15}", end="")
    for lbl in ["Healthy", "COPD", "Non-COPD"]:
        print(f" {lbl:>10}", end="")
    print(f" {'Total':>10}")
    print("-" * 50)
    for src in sorted(source_counter.keys()):
        src_records = [r for r in records if r["source"] == src]
        src_label_cnt = Counter(r["label"] for r in src_records)
        print(f"  {src:<13}", end="")
        row_total = 0
        for lbl in ["Healthy", "COPD", "Non-COPD"]:
            cnt = src_label_cnt.get(lbl, 0)
            row_total += cnt
            print(f" {cnt:>10}", end="")
        print(f" {row_total:>10}")

    # Output paths
    print(f"\n{'─' * 70}")
    print(f"  Output directory : {output_dir}")
    print(f"  Labels CSV       : {output_dir / 'labels.csv'}")
    print(f"  Audio directory  : {output_dir / 'audio'}")
    for lbl in ["Healthy", "COPD", "Non-COPD"]:
        sub_dir = output_dir / "audio" / lbl
        if sub_dir.exists():
            n_files = len(list(sub_dir.iterdir()))
            print(f"    └── {lbl}/ : {n_files} files")
    print(f"{'─' * 70}")

    # Ghi summary vào file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Combined Respiratory Sound Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(records)}\n\n")
        f.write("3-Class Distribution:\n")
        for lbl in ["Healthy", "COPD", "Non-COPD"]:
            cnt = label_counter.get(lbl, 0)
            pct = cnt / len(records) * 100
            f.write(f"  {lbl}: {cnt} ({pct:.1f}%)\n")
        f.write(f"\nSource Distribution:\n")
        for src, cnt in sorted(source_counter.items()):
            f.write(f"  {src}: {cnt}\n")
        f.write(f"\nOriginal Labels:\n")
        for lbl, cnt in sorted(orig_counter.items(), key=lambda x: -x[1]):
            mapped = map_to_3class(lbl)
            f.write(f"  {lbl}: {cnt} → {mapped}\n")

    print(f"\n  ✅ Gộp dataset hoàn tất!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Gộp 2 dataset âm thanh hô hấp thành 3-class dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink", "skip"],
        default="symlink",
        help="Cách xử lý file audio: copy (copy thật), "
             "symlink (tạo symbolic link), skip (chỉ tạo CSV). "
             "Mặc định: symlink"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Thư mục output. Mặc định: {DEFAULT_OUTPUT_DIR}"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    combine_datasets(output_dir=output_dir, mode=args.mode)


if __name__ == "__main__":
    main()
