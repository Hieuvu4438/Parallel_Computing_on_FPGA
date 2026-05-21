import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─── Draw.io Pastel Colors ────────────────────────────────────────────────────
C_INNER  = "#dae8fc"; E_INNER  = "#6c8ebf"   # Nút quyết định (decision)
C_LEAF_0 = "#d5e8d4"; E_LEAF_0 = "#82b366"   # Lá lớp 0
C_LEAF_1 = "#fff2cc"; E_LEAF_1 = "#d6b656"   # Lá lớp 1
C_LEAF_2 = "#f8cecc"; E_LEAF_2 = "#b85450"   # Lá lớp 2
C_VOTE   = "#e1d5e7"; E_VOTE   = "#9673a6"   # Majority Voting
# ─────────────────────────────────────────────────────────────────────────────

LEAF_COLORS = [
    (C_LEAF_0, E_LEAF_0),
    (C_LEAF_1, E_LEAF_1),
    (C_LEAF_2, E_LEAF_2),
]

def draw_circle(ax, cx, cy, r, fc, ec, lw=1.6, zorder=4):
    circ = patches.Circle((cx, cy), r,
                           linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    ax.add_patch(circ)

def draw_edge(ax, x1, y1, x2, y2, r, color="#666666", lw=1.3):
    """Cạnh từ mép node cha → mép node con (cắt phần bán kính r)."""
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    sx, sy = x1 + ux * r, y1 + uy * r
    ex, ey = x2 - ux * r, y2 - uy * r
    ax.annotate(
        "", xy=(ex, ey), xytext=(sx, sy),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=10, shrinkA=0, shrinkB=0),
        zorder=3
    )

def draw_one_tree(ax, root_x, top_y, scale=1.0, r=0.22):
    """
    Vẽ cây nhị phân depth-3.  Trả về list (x,y) các lá và y thấp nhất.
    """
    dy = scale * 1.05   # khoảng cách dọc giữa hai mức

    # Level 0 – root
    lv0 = [(root_x, top_y)]
    draw_circle(ax, root_x, top_y, r, C_INNER, E_INNER)

    # Level 1
    sp1 = scale * 1.35
    lv1 = []
    for cx, cy in lv0:
        for s in [-1, 1]:
            nx, ny = cx + s * sp1, cy - dy
            lv1.append((nx, ny))
            draw_circle(ax, nx, ny, r, C_INNER, E_INNER)
            draw_edge(ax, cx, cy, nx, ny, r)

    # Level 2
    sp2 = scale * 0.62
    lv2 = []
    for cx, cy in lv1:
        for s in [-1, 1]:
            nx, ny = cx + s * sp2, cy - dy
            lv2.append((nx, ny))
            draw_circle(ax, nx, ny, r, C_INNER, E_INNER)
            draw_edge(ax, cx, cy, nx, ny, r)

    # Level 3 – leaves
    sp3 = scale * 0.28
    leaves = []
    for k, (cx, cy) in enumerate(lv2):
        for s in [-1, 1]:
            nx, ny = cx + s * sp3, cy - dy
            fc, ec = LEAF_COLORS[k % len(LEAF_COLORS)]
            draw_circle(ax, nx, ny, r * 0.78, fc, ec, lw=1.4)
            draw_edge(ax, cx, cy, nx, ny, r)
            leaves.append((nx, ny))

    return leaves

# ─────────────────────────────────────────────────────────────────────────────

def create_rf_diagram(output_path):
    N_TREES    = 3      # Số cây (hiển thị đại diện cho Septuple Forest)
    SCALE      = 0.68   # Scale nhỏ gọn cho khổ dọc
    R          = 0.20   # Bán kính node
    V_GAP      = 4.6    # Khoảng cách dọc giữa root của hai cây liền kề

    # Khổ DỌC (portrait): width nhỏ, height lớn
    fig, ax = plt.subplots(figsize=(5.0, N_TREES * V_GAP * 0.58 + 2.0))

    leaf_reprs = []   # (x_centroid, y_leaf_bottom) mỗi cây

    for i in range(N_TREES):
        root_x = 0.0
        root_y = -i * V_GAP      # Cây 0 cao nhất, cây N-1 thấp nhất

        leaves = draw_one_tree(ax, root_x, root_y, scale=SCALE, r=R)

        cx_c  = np.mean([l[0] for l in leaves])
        cy_bot = min(l[1] for l in leaves)
        leaf_reprs.append((cx_c, cy_bot))

    # ── Majority Voting node ở đáy ──────────────────────────────────────────
    vote_x = 0.0
    vote_y = leaf_reprs[-1][1] - SCALE * 1.7
    vote_r = 0.33
    draw_circle(ax, vote_x, vote_y, vote_r, C_VOTE, E_VOTE, lw=2.2)

    for cx, cy in leaf_reprs:
        draw_edge(ax, cx, cy, vote_x, vote_y, R * 0.45,
                  color="#9673a6", lw=1.6)

    # Cài đặt khung nhìn
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(vote_y - 0.7, 0.65)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"-> Đã tạo biểu đồ RF (dọc) tại: {output_path}")

if __name__ == "__main__":
    out_dir = "./visualizations"
    os.makedirs(out_dir, exist_ok=True)
    create_rf_diagram(os.path.join(out_dir, "symbolic_rf_architecture.png"))
