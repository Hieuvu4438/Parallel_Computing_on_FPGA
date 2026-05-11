import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_prism(ax, x, y, width, height, depth, facecolor, edgecolor='black', alpha=0.85):
    """
    Vẽ một khối hộp 3D (Prism) giả lập trên trục 2D.
    width: Số lượng channels (Chiều ngang)
    height: Spatial Height (Chiều cao)
    depth: Spatial Width (Chiều sâu nghiêng)
    """
    # Góc nghiêng cho cảm giác 3D (Isometric projection offset)
    dx = depth * 0.4
    dy = depth * 0.4

    # Điều chỉnh màu cho các mặt để tạo bóng (shading) 3D
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(facecolor)
    top_color = tuple(min(1.0, c * 1.2) for c in rgb)   # Sáng hơn
    right_color = tuple(max(0.0, c * 0.8) for c in rgb) # Tối hơn

    # Mặt trước (Front)
    front = patches.Rectangle((x, y), width, height, linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha, zorder=2)
    ax.add_patch(front)

    # Mặt trên (Top)
    top = patches.Polygon([
        [x, y + height],
        [x + width, y + height],
        [x + width + dx, y + height + dy],
        [x + dx, y + height + dy]
    ], closed=True, linewidth=1.2, edgecolor=edgecolor, facecolor=top_color, alpha=alpha, zorder=1)
    ax.add_patch(top)

    # Mặt phải (Right)
    right = patches.Polygon([
        [x + width, y],
        [x + width + dx, y + dy],
        [x + width + dx, y + height + dy],
        [x + width, y + height]
    ], closed=True, linewidth=1.2, edgecolor=edgecolor, facecolor=right_color, alpha=alpha, zorder=1)
    ax.add_patch(right)

def create_cnn_diagram(output_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Định nghĩa các layer mang tính biểu tượng (dựa trên cấu trúc MobileNetV2 / CNN thông thường)
    # Cấu trúc: [Tên (trống), Channels (Width), Spatial H, Spatial W (Depth), Color]
    # Bảng màu pastel chuẩn draw.io (xanh nhạt, xanh lá nhạt, vàng nhạt, cam nhạt, hồng nhạt, tím nhạt, xám nhạt)
    layers = [
        {"name": "", "c": 0.5, "h": 12.0, "w": 12.0, "color": "#dae8fc"},
        {"name": "", "c": 1.5, "h": 9.0,  "w": 9.0,  "color": "#d5e8d4"},
        {"name": "", "c": 2.5, "h": 6.5,  "w": 6.5,  "color": "#fff2cc"},
        {"name": "", "c": 4.0, "h": 4.5,  "w": 4.5,  "color": "#ffe6cc"},
        {"name": "", "c": 6.0, "h": 3.0,  "w": 3.0,  "color": "#f8cecc"},
        {"name": "", "c": 6.0, "h": 0.8,  "w": 0.8,  "color": "#e1d5e7"},
        {"name": "", "c": 1.0, "h": 5.0,  "w": 0.5,  "color": "#f5f5f5"}
    ]

    current_x = 0
    gap = 0.3  # Khoảng cách giữa các layer cực sát nhau

    # Vẽ từng layer
    for i, layer in enumerate(layers):
        # Canh giữa các block theo trục Y
        y_center_offset = - (layer["h"] / 2.0)
        
        draw_prism(ax, current_x, y_center_offset, 
                   width=layer["c"], height=layer["h"], depth=layer["w"], 
                   facecolor=layer["color"])
        
        # Bỏ đi mũi tên kết nối và Text theo yêu cầu
        
        # Cập nhật tọa độ X cho layer tiếp theo
        current_x += layer["c"] + gap

    # Cài đặt hiển thị
    ax.set_xlim(-2, current_x + 2)
    ax.set_ylim(-12, 10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"-> Đã tạo biểu đồ CNN tại: {output_path}")

if __name__ == "__main__":
    out_dir = "./visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "symbolic_cnn_architecture.png")
    create_cnn_diagram(out_path)
