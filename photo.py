import matplotlib.pyplot as plt
import os

# 设置 IEEE 期刊风格的绘图参数
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

def plot_comparison():
    # --- 1. 硬编码数据 (Manual Data Entry) ---
    
    # 关键修改：恢复 X 轴绝对坐标，全部对齐到 50
    x_ann = [0, 25, 50]
    x_snn = [0, 25, 50]
    
    # Y 轴数据 (Accuracy %)
    y_ann = [96.22, 94.96, 92.90] 
    y_snn = [94.10, 94.45, 92.84] 

    # 2. 绘图
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # 绘制 SNN 曲线 (蓝实线, 方块) - 放在底层 (zorder=5)
    # 方块稍微大一点 (markersize=9)，作为背景
    line2, = ax.plot(x_snn, y_snn, 
            marker='s', markersize=9, linestyle='-', color='#1f77b4', label='FW-SNN (Ours)', zorder=5)

    # 绘制 ANN 曲线 (红虚线, 圆圈) - 放在上层 (zorder=10)
    # 关键修改：使用空心圆圈 (markerfacecolor='white')
    # 这样即使重叠，圆圈中间是白色的，能透出下面蓝方块的边角，视觉上暗示“重叠”
    line1, = ax.plot(x_ann, y_ann, 
            marker='o', markersize=6, linestyle='--', color='#d62728', 
            markerfacecolor='white', markeredgewidth=2,
            label='VGG9-CNN (ANN)', zorder=10)

    # 3. 美化图表
    ax.set_xlabel('Pruning Ratio (Frequency Dimensions Removed)', fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    ax.set_ylim(92.0, 97.5)
    ax.set_xticks([0, 25, 50])
    ax.set_xticklabels(['0%', '25%', '50%'])
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.legend(handles=[line2, line1], loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, edgecolor='black', fancybox=False) 

    # --- 4. 定制化标签位置 (Customized Labels) ---
    
    # ================= ANN (红色) 标签 =================
    ax.annotate(f'{y_ann[0]:.2f}%', (x_ann[0], y_ann[0]), 
                textcoords="offset points", xytext=(0, 8), ha='center', color='#d62728', fontsize=10)
    
    ax.annotate(f'{y_ann[1]:.2f}%', (x_ann[1], y_ann[1]), 
                textcoords="offset points", xytext=(0, 8), ha='center', color='#d62728', fontsize=10)
    
    # Point 3 (50%): 92.90% -> 往右上移动
    # xytext=(10, 10): 向右偏10，向上偏10
    # ha='left': 左对齐，文字向右延伸
    ax.annotate(f'{y_ann[2]:.2f}%', (x_ann[2], y_ann[2]), 
                textcoords="offset points", xytext=(-20, 10), ha='left', color='#d62728', fontsize=10)

    # ================= SNN (蓝色) 标签 =================
    ax.annotate(f'{y_snn[0]:.2f}%', (x_snn[0], y_snn[0]), 
                textcoords="offset points", xytext=(0, -15), ha='center', color='#1f77b4', weight='bold', fontsize=10)
    
    ax.annotate(f'{y_snn[1]:.2f}%', (x_snn[1], y_snn[1]), 
                textcoords="offset points", xytext=(0, -15), ha='center', color='#1f77b4', weight='bold', fontsize=10)
    
    # Point 3 (50%): 92.84% -> 往左下移动
    # xytext=(-15, -15): 向左偏15，向下偏15
    # ha='right': 右对齐，文字向左延伸
    ax.annotate(f'{y_snn[2]:.2f}%', (x_snn[2], y_snn[2]), 
                textcoords="offset points", xytext=(20, -15), ha='right', color='#1f77b4', weight='bold', fontsize=10)

    # 5. 保存图片
    save_path = 'snn_vs_ann_comparison.png'
    save_path_pdf = 'snn_vs_ann_comparison.pdf'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"Plot saved to {os.path.abspath(save_path)}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()