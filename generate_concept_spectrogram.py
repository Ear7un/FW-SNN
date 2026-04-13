import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

def generate_concept_spectrogram():
    # --- 1. 参数设置 ---
    width = 400   # 时间维度 (Time)
    height = 200  # 频率维度 (Frequency)
    
    # 创建空白画布
    spectrogram = np.zeros((height, width))
    
    # 生成时间轴
    t = np.linspace(0, 10, width)
    
    # --- 2. 下半部分：构建清晰的“信号带” (Low Frequency Signal) ---
    # 模拟引擎声或人声基频：几条明显的横向波纹
    # 我们在 0-40% 的高度范围内生成信号
    num_harmonics = 4
    base_freq_indices = [20, 45, 70, 90] # 在高度上的位置
    
    for i, y_idx in enumerate(base_freq_indices):
        # 基础强度
        intensity = 0.8 + 0.2 * np.random.rand()
        
        # 添加一点波动 (Wiggle)，让它看起来像真实的音频而不是直线
        wiggle = 5 * np.sin(t * (1.5 + 0.1*i)) 
        
        # 在每一列填入波纹
        for x in range(width):
            # 计算当前时间点的波纹中心
            center_y = int(y_idx + wiggle[x])
            if 0 <= center_y < height:
                # 在纵向添加一个高斯分布的能量，让线条有厚度
                for dy in range(-8, 9):
                    curr_y = center_y + dy
                    if 0 <= curr_y < height:
                        # 距离中心越近越亮
                        falloff = np.exp(-(dy**2) / (2 * 2**2))
                        spectrogram[curr_y, x] += intensity * falloff

    # --- 3. 上半部分：构建杂乱的“噪声带” (High Frequency Noise) ---
    # 在 50%-100% 的高度范围内生成噪声
    noise_start_h = int(height * 0.5)
    
    # 生成高斯白噪声
    noise = np.random.randn(height - noise_start_h, width)
    
    # 对噪声进行轻微模糊，让它看起来像“风声”或“纹理”，而不是电视雪花
    noise = gaussian_filter(noise, sigma=1.5)
    
    # 归一化噪声并调整强度
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise *= 0.4  # 噪声强度通常比信号弱
    
    # 将噪声叠加到频谱图上部
    spectrogram[noise_start_h:, :] += noise
    
    # --- 4. 全局润色 ---
    # 添加一些随机底噪到底部，避免太假
    global_noise = np.random.randn(height, width) * 0.05
    spectrogram += global_noise
    
    # 限制数值范围
    spectrogram = np.clip(spectrogram, 0, 1)

    # --- 5. 绘图 ---
    # 设置 IEEE 风格
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 修改：使用 'PuBu' (Purple-Blue) 色图
    # 效果：0.0(背景)为白色，0.4(噪声)为浅蓝，1.0(信号)为深蓝/紫
    # 这样在白纸/白PPT上非常清爽，且符合“信号深、噪声浅”的直觉
    cax = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='PuBu')
    
    # 标注轴
    ax.set_ylabel('Frequency ($F$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time ($T$)', fontsize=14, fontweight='bold')
    
    # 去掉刻度数字（Schematic diagram 不需要具体数值）
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加视觉辅助线（可选，用于自己看）
    # ax.axhline(y=noise_start_h, color='white', linestyle='--', alpha=0.3)
    # ax.text(10, height - 20, 'Noise Dominated', color='white', alpha=0.7)
    # ax.text(10, 30, 'Signal Dominated', color='white', alpha=0.7)

    plt.tight_layout()
    
    # 保存
    save_path = 'concept_spectrogram.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Spectrogram saved to {save_path}")
    
    # --- 额外：生成带透明背景的图，方便 PPT 叠加 ---
    fig_transparent = plt.figure(figsize=(6, 4))
    ax_t = fig_transparent.add_axes([0, 0, 1, 1])
    ax_t.imshow(spectrogram, aspect='auto', origin='lower', cmap='inferno')
    ax_t.axis('off') # 只要图，不要轴
    plt.savefig('concept_spectrogram_transparent.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Transparent version saved.")

    plt.show()

if __name__ == "__main__":
    generate_concept_spectrogram()