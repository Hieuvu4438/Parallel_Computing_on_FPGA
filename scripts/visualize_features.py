import os
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def extract_and_plot_features(audio_path, output_dir):
    """
    Trích xuất RMS, ZCR, MFCC 39D từ audio và lưu mỗi đặc trưng thành 1 ảnh riêng biệt.
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy tên file gốc (không bao gồm đuôi file)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    print(f"Đang xử lý file: {audio_path}")
    
    # Cấu hình theo bài báo: Sample rate 4000 Hz
    sr_target = 4000
    y, sr = librosa.load(audio_path, sr=sr_target)
    print(f"Đã load audio với sample rate: {sr} Hz")
    
    # Thiết lập frame_length và hop_length cho audio hô hấp
    frame_length = 512  # Khoảng 128ms tại 4000Hz
    hop_length = 256    # Khoảng 64ms
    
    # ---------------------------------------------------------
    # 1. RMS Energy
    # ---------------------------------------------------------
    print("Đang trích xuất RMS Energy...")
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    
    plt.figure(figsize=(10, 4))
    plt.plot(times, rms[0], color='#1f77b4', linewidth=1.5)  # Màu xanh dương (chuẩn tab:blue)
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    rms_path = os.path.join(output_dir, f"{base_name}_rms.png")
    plt.savefig(rms_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    print(f" -> Đã lưu ảnh RMS tại: {rms_path}")
    
    # ---------------------------------------------------------
    # 2. Zero Crossing Rate (ZCR)
    # ---------------------------------------------------------
    print("Đang trích xuất Zero Crossing Rate (ZCR)...")
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)
    
    plt.figure(figsize=(10, 4))
    plt.plot(times, zcr[0], color='#ff7f0e', linewidth=1.5)  # Màu cam (chuẩn tab:orange)
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    zcr_path = os.path.join(output_dir, f"{base_name}_zcr.png")
    plt.savefig(zcr_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    print(f" -> Đã lưu ảnh ZCR tại: {zcr_path}")
    
    # ---------------------------------------------------------
    # 3. MFCC 39D (13 Static + 13 Delta + 13 Delta-Delta)
    # ---------------------------------------------------------
    print("Đang trích xuất MFCC 39D...")
    # Tính 13 hệ số tĩnh (Static)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    
    # Tính 13 hệ số bậc 1 (Delta)
    delta_mfcc = librosa.feature.delta(mfcc)
    
    # Tính 13 hệ số bậc 2 (Delta-Delta)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Nối chúng lại thành vector 39 chiều (MFCC 39D)
    mfcc_39d = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    
    plt.figure(figsize=(12, 6))
    # Sử dụng cmap='jet' cho hiển thị phổ sáng, có độ tương phản rực rỡ chuẩn Computer Vision cổ điển
    librosa.display.specshow(mfcc_39d, sr=sr, hop_length=hop_length, cmap='jet')
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    mfcc_path = os.path.join(output_dir, f"{base_name}_mfcc39d.png")
    plt.savefig(mfcc_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f" -> Đã lưu ảnh MFCC 39D tại: {mfcc_path}")
    
    print("\nHoàn tất quá trình trích xuất và visualize!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script tạo 3 ảnh visualization (RMS, ZCR, MFCC 39D) từ 1 file audio.")
    parser.add_argument("--audio", type=str, required=True, help="Đường dẫn đến file audio đầu vào (.wav)")
    parser.add_argument("--outdir", type=str, default="./visualizations", help="Thư mục lưu 3 file ảnh đầu ra (mặc định: ./visualizations)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"[Lỗi] Không tìm thấy file audio tại: {args.audio}")
        exit(1)
        
    extract_and_plot_features(args.audio, args.outdir)
