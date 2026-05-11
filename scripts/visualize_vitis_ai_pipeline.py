"""
Visualize the Vitis AI deployment pipeline for MobileNetV2 on Ultra96-V2 (Zynq UltraScale+).

Pipeline (theo UG1414 Vitis AI 3.5):
  [MobileNetV2 FP32]
       ↓
  [Vitis AI Quantizer]  (vai_q_pytorch / INT8 PTQ/QAT)
       ↓
  [Quantized .xmodel]
       ↓
  [Vitis AI Compiler]  (vai_c_xir → DPU target)
       ↓
  [Compiled .xmodel]
       ──────── FPGA (PL/PS) ──────────
       ↓
  ┌──────────────────────────────────┐
  │          Ultra96-V2              │
  │  PS (ARM Cortex-A53)             │
  │  ├── VART Runtime (C++/Python)   │
  │  └── Vitis AI Library            │
  │                                  │
  │  PL (Programmable Logic)         │
  │  └── DPU (DPUCZDX8G)            │
  │      ├── Conv Engine             │
  │      ├── Activation Engine       │
  │      ├── Pooling Engine          │
  │      └── Misc Engine             │
  └──────────────────────────────────┘
       ↓
  [Output: Class Prediction]
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Draw.io / AMD Xilinx color palette ────────────────────────────────────────
# Xilinx/AMD brand colours: đỏ đậm #E8003D, cam #FF6600; pastel draw.io dùng cho sub-blocks.
C_TOOL     = "#dae8fc"; E_TOOL    = "#6c8ebf"   # Vitis AI tools (xanh nhạt)
C_ARTIFACT = "#d5e8d4"; E_ARTIFACT= "#82b366"   # File artifacts (.pt, .xmodel)
C_PS       = "#ffe6cc"; E_PS      = "#d79b00"   # PS (ARM) – cam nhạt
C_PL       = "#f8cecc"; E_PL      = "#b85450"   # PL overall – hồng nhạt
C_DPU      = "#e1d5e7"; E_DPU     = "#9673a6"   # DPU block – tím nhạt
C_ENGINE   = "#fff2cc"; E_ENGINE  = "#d6b656"   # Các engine con – vàng nhạt
C_OUT      = "#d5e8d4"; E_OUT     = "#82b366"   # Output – xanh lá
C_BG_FPGA  = "#f5f5f5"; E_BG_FPGA = "#cccccc"   # FPGA container background

FONT_MAIN  = dict(fontsize=9.5, fontweight='bold', ha='center', va='center', color='#333333')
FONT_SUB   = dict(fontsize=8.0, ha='center', va='center', color='#555555')
ARROW_KW   = dict(arrowstyle='-|>', color='#444444', lw=1.6, mutation_scale=13)

def fbox(ax, cx, cy, w, h, fc, ec, lw=1.6, radius=0.18, zorder=3):
    """Fancy rounded box (flat UI style)."""
    b = patches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder
    )
    ax.add_patch(b)

def arrow(ax, x1, y1, x2, y2, zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(**ARROW_KW), zorder=zorder)


def create_vitis_ai_diagram(output_path):
    fig, ax = plt.subplots(figsize=(8.0, 14.0))

    # ── Layout constants ──────────────────────────────────────────────────────
    CX   = 4.0          # center X của toàn bộ diagram
    BW   = 5.0          # chiều rộng block chính
    BH   = 0.70         # chiều cao block chính
    STEP = 1.55         # bước dịch Y giữa các block

    # ── 1. Input Model ────────────────────────────────────────────────────────
    y = 13.0
    fbox(ax, CX, y, BW, BH, C_ARTIFACT, E_ARTIFACT)
    ax.text(CX, y, "MobileNetV2  (FP32, PyTorch)", **FONT_MAIN)

    # ── 2. Vitis AI Quantizer ─────────────────────────────────────────────────
    y -= STEP
    arrow(ax, CX, 13.0 - BH/2, CX, y + BH/2)
    fbox(ax, CX, y, BW, BH, C_TOOL, E_TOOL)
    ax.text(CX, y + 0.14, "Vitis AI Quantizer", **FONT_MAIN)
    ax.text(CX, y - 0.18, "vai_q_pytorch  |  INT8 PTQ/QAT", **FONT_SUB)

    # ── 3. Quantized .xmodel ──────────────────────────────────────────────────
    y -= STEP
    arrow(ax, CX, y + STEP - BH/2, CX, y + BH/2)
    fbox(ax, CX, y, BW, BH, C_ARTIFACT, E_ARTIFACT)
    ax.text(CX, y, "Quantized Model  (.xmodel / INT8)", **FONT_MAIN)

    # ── 4. Vitis AI Compiler ──────────────────────────────────────────────────
    y -= STEP
    arrow(ax, CX, y + STEP - BH/2, CX, y + BH/2)
    fbox(ax, CX, y, BW, BH, C_TOOL, E_TOOL)
    ax.text(CX, y + 0.14, "Vitis AI Compiler", **FONT_MAIN)
    ax.text(CX, y - 0.18, "vai_c_xir  |  Target: DPUCZDX8G (Ultra96-V2)", **FONT_SUB)

    # ── 5. Compiled .xmodel ───────────────────────────────────────────────────
    y -= STEP
    arrow(ax, CX, y + STEP - BH/2, CX, y + BH/2)
    fbox(ax, CX, y, BW, BH, C_ARTIFACT, E_ARTIFACT)
    ax.text(CX, y, "Compiled DPU Subgraph  (.xmodel)", **FONT_MAIN)

    compiled_y = y   # lưu lại để vẽ mũi tên vào FPGA box

    # ── 6. FPGA Container (Ultra96-V2) ────────────────────────────────────────
    fpga_top    = compiled_y - STEP * 0.6
    fpga_bottom = 0.2
    fpga_height = fpga_top - fpga_bottom
    fpga_cy     = (fpga_top + fpga_bottom) / 2

    arrow(ax, CX, compiled_y - BH/2, CX, fpga_top)

    # Outer FPGA box
    fbox(ax, CX, fpga_cy, BW + 1.2, fpga_height, C_BG_FPGA, E_BG_FPGA,
         lw=2.2, radius=0.30, zorder=1)
    ax.text(CX, fpga_top - 0.25, "Ultra96-V2  (Zynq UltraScale+ MPSoC)",
            fontsize=9, fontweight='bold', ha='center', va='center',
            color='#555555', style='italic', zorder=4)

    # ── 6a. PS sub-box ────────────────────────────────────────────────────────
    ps_cx = CX
    ps_cy = fpga_top - 1.20
    ps_w, ps_h = 4.6, 1.35
    fbox(ax, ps_cx, ps_cy, ps_w, ps_h, C_PS, E_PS, lw=1.6, radius=0.20, zorder=2)
    ax.text(ps_cx, ps_cy + 0.34, "PS  –  ARM Cortex-A53", **FONT_MAIN, zorder=5)
    # Sub-items
    for i, txt in enumerate(["VART Runtime (C++ / Python)", "Vitis AI Library"]):
        ax.text(ps_cx, ps_cy + 0.00 - i*0.32, "• " + txt,
                fontsize=8.2, ha='center', va='center', color='#444444', zorder=5)

    # ── 6b. PL sub-box (DPU) ─────────────────────────────────────────────────
    pl_cy = ps_cy - ps_h/2 - 0.10 - 1.55
    pl_w, pl_h = 4.6, 3.10
    fbox(ax, ps_cx, pl_cy, pl_w, pl_h, C_PL, E_PL, lw=1.6, radius=0.20, zorder=2)
    ax.text(ps_cx, pl_cy + pl_h/2 - 0.28, "PL  –  Programmable Logic",
            **FONT_MAIN, zorder=5)

    # DPU block inside PL
    dpu_cy = pl_cy - 0.05
    dpu_w, dpu_h = 3.8, 2.20
    fbox(ax, ps_cx, dpu_cy, dpu_w, dpu_h, C_DPU, E_DPU, lw=1.4, radius=0.16, zorder=3)
    ax.text(ps_cx, dpu_cy + dpu_h/2 - 0.28, "DPU  (DPUCZDX8G)",
            **FONT_MAIN, zorder=5)

    # Engine sub-blocks inside DPU
    engines = ["Conv Engine", "Activation Engine", "Pooling Engine", "Misc Engine"]
    eng_w, eng_h = 1.52, 0.42
    cols = 2
    for idx, eng in enumerate(engines):
        col = idx % cols
        row = idx // cols
        ex = ps_cx - eng_w/2 - 0.06 + col * (eng_w + 0.12)
        ey = dpu_cy + 0.35 - row * (eng_h + 0.22)
        fbox(ax, ex, ey, eng_w, eng_h, C_ENGINE, E_ENGINE, lw=1.2, radius=0.12, zorder=4)
        ax.text(ex, ey, eng, fontsize=7.8, ha='center', va='center',
                color='#333333', fontweight='bold', zorder=5)

    # PS → PL arrow (vertical)
    arrow(ax, ps_cx, ps_cy - ps_h/2, ps_cx, pl_cy + pl_h/2)

    # ── 7. Output ─────────────────────────────────────────────────────────────
    out_y = fpga_bottom - STEP * 0.85
    arrow(ax, CX, fpga_bottom, CX, out_y + BH/2)
    fbox(ax, CX, out_y, BW, BH, C_OUT, E_OUT)
    ax.text(CX, out_y, "Output  –  Class Prediction (COPD / Healthy / Non-COPD)", **FONT_MAIN)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(0.8, 7.2)
    ax.set_ylim(out_y - 0.5, 13.5)
    ax.axis('off')

    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"-> Saved: {output_path}")


if __name__ == "__main__":
    out_dir = "./visualizations"
    os.makedirs(out_dir, exist_ok=True)
    create_vitis_ai_diagram(os.path.join(out_dir, "vitis_ai_fpga_pipeline.png"))
