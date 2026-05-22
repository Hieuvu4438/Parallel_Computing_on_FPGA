import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import PAPER_FIGURES_DIR

def draw_box(ax, x, y, width, height, text, bg_color, edge_color, text_color='black', fontsize=10):
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        ec=edge_color, fc=bg_color, lw=1.5, zorder=2
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            color=text_color, fontsize=fontsize, fontweight='bold', zorder=3)
    return (x + width/2, y + height/2), (x + width, y + height/2), (x, y + height/2), (x + width/2, y), (x + width/2, y + height)

def draw_arrow(ax, start, end, style="->", connectionstyle="arc3", color='#555555'):
    ax.annotate("",
                xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle=style, color=color, lw=2.0,
                                connectionstyle=connectionstyle, shrinkA=0, shrinkB=0),
                zorder=1)

def plot_kd_diagram(filename):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.2, 1.1)
    ax.axis('off')
    
    # Colors
    c_input = '#E1F5FE'
    c_teacher = '#FFF3E0'
    c_student = '#E8F5E9'
    c_op = '#F3E5F5'
    c_loss = '#FFEBEE'
    
    e_input = '#0288D1'
    e_teacher = '#F57C00'
    e_student = '#388E3C'
    e_op = '#7B1FA2'
    e_loss = '#D32F2F'

    # Box dimensions
    w, h = 0.16, 0.1
    
    # Coordinates
    # Input
    pos_input = (-0.05, 0.45)
    c_inp, right_inp, left_inp, bot_inp, top_inp = draw_box(ax, pos_input[0], pos_input[1], w, h, "Input\n(Hybrid Spec)", c_input, e_input)
    
    # Models
    pos_teacher = (0.2, 0.75)
    c_t, right_t, left_t, bot_t, top_t = draw_box(ax, pos_teacher[0], pos_teacher[1], w, h, "Teacher Ensemble\n(EfficientNet-B0)", c_teacher, e_teacher)
    
    pos_student = (0.2, 0.15)
    c_s, right_s, left_s, bot_s, top_s = draw_box(ax, pos_student[0], pos_student[1], w, h, "Student\n(MobileNetV2)", c_student, e_student)
    
    # Logits
    pos_t_logit = (0.45, 0.75)
    c_tl, right_tl, left_tl, bot_tl, top_tl = draw_box(ax, pos_t_logit[0], pos_t_logit[1], w-0.04, h, "Logits", c_op, e_op)
    
    pos_s_logit = (0.45, 0.15)
    c_sl, right_sl, left_sl, bot_sl, top_sl = draw_box(ax, pos_s_logit[0], pos_s_logit[1], w-0.04, h, "Logits", c_op, e_op)
    
    # Softmax / Temp
    pos_t_soft = (0.65, 0.75)
    c_tsf, right_tsf, left_tsf, bot_tsf, top_tsf = draw_box(ax, pos_t_soft[0], pos_t_soft[1], w, h, "Softmax (T=4.0)\nSoft Labels", c_op, e_op)
    
    pos_s_soft = (0.65, 0.35)
    c_ssf, right_ssf, left_ssf, bot_ssf, top_ssf = draw_box(ax, pos_s_soft[0], pos_s_soft[1], w, h, "Softmax (T=4.0)\nSoft Preds", c_op, e_op)
    
    pos_s_hard = (0.65, 0.15)
    c_shd, right_shd, left_shd, bot_shd, top_shd = draw_box(ax, pos_s_hard[0], pos_s_hard[1], w, h, "Softmax (T=1.0)\nHard Preds", c_op, e_op)
    
    # True Label
    pos_label = (0.65, -0.05)
    c_lab, right_lab, left_lab, bot_lab, top_lab = draw_box(ax, pos_label[0], pos_label[1], w, h, "True Labels", c_input, e_input)
    
    # Losses
    pos_kd_loss = (0.88, 0.55)
    c_kdl, right_kdl, left_kdl, bot_kdl, top_kdl = draw_box(ax, pos_kd_loss[0], pos_kd_loss[1], w, h, "KD Loss\n(KL Divergence)", c_loss, e_loss)
    
    pos_hd_loss = (0.88, 0.05)
    c_hdl, right_hdl, left_hdl, bot_hdl, top_hdl = draw_box(ax, pos_hd_loss[0], pos_hd_loss[1], w, h, "Hard Loss\n(Focal Loss)", c_loss, e_loss)
    
    # Total Loss
    pos_tot_loss = (1.05, 0.3)
    c_tot, right_tot, left_tot, bot_tot, top_tot = draw_box(ax, pos_tot_loss[0], pos_tot_loss[1], w-0.04, h, "Total Loss\n(\u03B1=0.7)", c_loss, e_loss)
    
    # ---- Draw Arrows ----
    # Input to Models
    draw_arrow(ax, right_inp, left_t, connectionstyle="arc3,rad=0.1")
    draw_arrow(ax, right_inp, left_s, connectionstyle="arc3,rad=-0.1")
    
    # Models to Logits
    draw_arrow(ax, right_t, left_tl)
    draw_arrow(ax, right_s, left_sl)
    
    # Logits to Softmax
    draw_arrow(ax, right_tl, left_tsf)
    draw_arrow(ax, right_sl, left_ssf, connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, right_sl, left_shd)
    
    # Softmax to KD Loss
    draw_arrow(ax, right_tsf, left_kdl, connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, right_ssf, left_kdl, connectionstyle="arc3,rad=0.1")
    
    # Hard Pred & True Label to Hard Loss
    draw_arrow(ax, right_shd, left_hdl, connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, right_lab, left_hdl, connectionstyle="arc3,rad=0.1")
    
    # Losses to Total Loss
    draw_arrow(ax, right_kdl, left_tot, connectionstyle="arc3,rad=0.1")
    draw_arrow(ax, right_hdl, left_tot, connectionstyle="arc3,rad=-0.1")
    
    # Add Multiplier text for total loss
    ax.text(0.96, 0.61, "x \u03B1", fontsize=10, fontweight='bold', color='#D32F2F')
    ax.text(0.96, 0.11, "x (1-\u03B1)", fontsize=10, fontweight='bold', color='#D32F2F')
    ax.text(0.68, 0.86, "Frozen", fontsize=10, fontweight='bold', color='#F57C00', style='italic')
    ax.text(0.68, 0.26, "Trainable", fontsize=10, fontweight='bold', color='#388E3C', style='italic')

    plt.tight_layout()
    plt.savefig(filename, format='png', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(filename.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot knowledge distillation architecture")
    parser.add_argument('--output_dir', type=str, default=str(PAPER_FIGURES_DIR))
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "KD_Architecture.png")
    plot_kd_diagram(out_file)
    print(f"Knowledge Distillation Architecture plot saved to: {out_file}")
