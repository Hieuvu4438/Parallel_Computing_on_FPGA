import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.common.paths import PAPER_FIGURES_DIR

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotlib.
    '''
    n_layers = len(layer_sizes)
    # Tinh toan khoang cach giua cac node de khong bi chong cheo
    max_nodes = max(layer_sizes)
    v_spacing = (top - bottom) / float(max_nodes + 1)
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    
    # Colors for IEEE standard (Clean, professional Draw.io style)
    # Input: Soft Blue, Hidden: Soft Amber, Output: Soft Green
    colors = ['#E3F2FD', '#FFF8E1', '#E8F5E9']
    edge_colors = ['#1565C0', '#FF8F00', '#2E7D32']
    
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], 
                                  c='#BDBDBD', lw=1.2, zorder=1, alpha=0.7)
                ax.add_artist(line)

    # Nodes
    # Dieu chinh ban kinh node de phu hop voi so luong node lon
    circle_radius = v_spacing / 3.0
    if circle_radius > 0.05:
        circle_radius = 0.05

    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        fill_color = colors[n % len(colors)]
        edge_color = edge_colors[n % len(edge_colors)]
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), circle_radius,
                                color=fill_color, ec=edge_color, lw=2.5, zorder=4)
            ax.add_artist(circle)

def generate_nn_image(layer_sizes, filename):
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.gca()
    ax.axis('off')
    
    draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, layer_sizes)
    
    plt.tight_layout()
    # Save as high-res PNG
    plt.savefig(filename, format='png', bbox_inches='tight', transparent=True, dpi=300)
    # Save as PDF (Vector graphic is often required for IEEE)
    plt.savefig(filename.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot teacher/student neural-network sketches")
    parser.add_argument('--output_dir', type=str, default=str(PAPER_FIGURES_DIR))
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Teacher Model: 3 layers, dense hidden layer
    # Input: 8 nodes, Hidden: 12 nodes, Output: 4 nodes
    teacher_layers = [8, 12, 4]
    generate_nn_image(teacher_layers, os.path.join(output_dir, "Teacher_Model.png"))
    print(f"Teacher model plotted and saved to: {os.path.join(output_dir, 'Teacher_Model.png')}")
    
    # Student Model: 3 layers, sparse hidden layer
    # Input: 8 nodes, Hidden: 5 nodes, Output: 4 nodes
    student_layers = [8, 5, 4]
    generate_nn_image(student_layers, os.path.join(output_dir, "Student_Model.png"))
    print(f"Student model plotted and saved to: {os.path.join(output_dir, 'Student_Model.png')}")
