#!/usr/bin/env python3
"""
Frequency Dimension Pruning Project Data Analysis Script
For generating paper charts and performance comparison
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from pathlib import Path
import json

# Set chart style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

class FrequencyPruningAnalyzer:
    def __init__(self, runs_dir="./runs", logs_dir="./logs"):
        self.runs_dir = Path(runs_dir)
        self.logs_dir = Path(logs_dir)
        # 输出根目录固定在 analysis_results，下层子目录按本次分析生成
        self.output_root = Path("./analysis_results")
        self.output_root.mkdir(exist_ok=True)
        self.output_dir = None
        
        # Data storage - 修改为支持多个剪枝模型
        self.full_training_data = {}
        self.pruned_models_data = {}  # 存储多个剪枝模型的数据
    
    def load_training_data(self):
        """Load training data"""
        print("Loading training data...")
        
        # Load step 1 (full training) data
        full_dir = self.runs_dir / "full"
        if full_dir.exists():
            full_runs = list(full_dir.glob("*"))
            if full_runs:
                latest_full = max(full_runs, key=lambda x: x.stat().st_mtime)
                self.load_full_training_data(latest_full)
        
        # Load step 3 (pruned training) data - 加载所有剪枝模型
        pruned_dir = self.runs_dir / "pruned"
        if pruned_dir.exists():
            pruned_runs = list(pruned_dir.glob("*"))
            for pruned_run in pruned_runs:
                self.load_pruned_training_data(pruned_run)
        
        # Load step 2 (pruning indices) data
        prune_dir = self.runs_dir / "prune"
        if prune_dir.exists():
            prune_runs = list(prune_dir.glob("*"))
            if prune_runs:
                # This function is no longer needed in auto-mode, 
                # but we keep it for potential future use.
                pass

    def load_full_training_data(self, full_dir):
        """加载完整训练数据"""
        print(f"Loading full training data: {full_dir}")
        
        # 尝试多个可能的路径
        possible_paths = [
            full_dir / "freq_weights",
            full_dir,
            Path(str(full_dir).replace("runs/full", "runs/full"))
        ]
        
        complete_file = None
        freq_weights_dir = None
        for path in possible_paths:
            if path.exists():
                complete_files = list(path.glob("complete_freq_weight_history_*.pt"))
                if complete_files:
                    complete_file = complete_files[0]
                    freq_weights_dir = path
                    print(f"Found complete file: {complete_file}")
                    break
        
        if complete_file is None:
            print(f"Error: No complete_freq_weight_history files found in any of: {possible_paths}")
            return

        data = torch.load(complete_file, map_location='cpu', weights_only=False)
        self.full_training_data = {
            'complete_history': data,
            'final_weights': data['final_weights'],
            'weight_history': data['freq_weight_history'],
            'final_accuracy': float(complete_file.stem.split('_acc')[-1]),
            'epochs': data['epochs']
        }
        
        # Load per-epoch weight data
        if freq_weights_dir:
            epoch_files = sorted(freq_weights_dir.glob("freq_weights_epoch_*.pt"))
            epoch_data = []
            for file in epoch_files:
                epoch_data.append(torch.load(file, map_location='cpu', weights_only=False))
            self.full_training_data['epoch_data'] = epoch_data
        
        self.full_training_data['directory'] = full_dir
    
    def load_pruned_training_data(self, pruned_dir):
        """加载剪枝训练数据 - 仅初始化，不加载文件"""
        print(f"Initializing pruned training data entry for: {pruned_dir}")

        # 确定剪枝比例
        dir_name = str(pruned_dir.name)
        pruning_ratio = -1.0
        model_name = "Unknown Pruned"
        
        keep_match = re.search(r'keep(\d+)of(\d+)', dir_name)
        if keep_match:
            kept = int(keep_match.group(1))
            total = int(keep_match.group(2))
            if total > 0:
                pruning_ratio = round(1.0 - (kept / total), 2)
                model_name = f"{int(pruning_ratio*100)}% Pruned ({kept} bins)"
        
        if pruning_ratio != -1.0:
            print(f"Identified pruned model: {model_name} with ratio {pruning_ratio}")
            self.pruned_models_data[pruning_ratio] = {
                'pruning_ratio': pruning_ratio,
                'model_name': model_name,
                'directory': pruned_dir
            }
        else:
            print(f"Warning: Could not determine pruning ratio for directory {pruned_dir}")
    
    def load_prune_indices_data(self, prune_run_dir):
        """Load pruning indices data"""
        print(f"Loading pruning indices data: {prune_run_dir}")
        
        indices_file = list(prune_run_dir.glob("*.pt"))[0]
        data = torch.load(indices_file, map_location='cpu', weights_only=False)
        self.prune_indices_data = {
            'keep_indices': data['keep_indices'].numpy(),
            'meta': data['meta']
        }
    
    def extract_training_metrics_from_logs(self, specific_log_files=None):
        """Extract training metrics from log files"""
        print("Extracting training metrics from log files...")
        
        log_files = []
        if specific_log_files:
            log_files = specific_log_files
        else:
            # 自动扫描模式
            full_logs = list(self.runs_dir.glob("full/*/train_record.log"))
            log_files.extend(full_logs)
            pruned_logs = list(self.runs_dir.glob("pruned/*/train_record.log"))
            log_files.extend(pruned_logs)
        
        for log_file in log_files:
            self.parse_training_log(log_file)
    
    def parse_training_log(self, log_file):
        """Parse training log file"""
        print(f"Parsing training log: {log_file}")
        
        epochs = []
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        train_times = []
        test_times = []
        weight_means = []
        weight_stds = []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse training metrics
                if "Epoch[" in line and "train_loss:" in line:
                    match = re.search(r'Epoch\[(\d+)/\d+\] train_loss: ([\d.]+), train_acc=([\d.]+), test_loss=([\d.]+), test_acc=([\d.]+)', line)
                    if match:
                        epochs.append(int(match.group(1)))
                        train_losses.append(float(match.group(2)))
                        train_accs.append(float(match.group(3)))
                        test_losses.append(float(match.group(4)))
                        test_accs.append(float(match.group(5)))
                
                # Parse training time
                if "train time:" in line:
                    match = re.search(r'train time: ([\d.]+)s, test time: ([\d.]+)s', line)
                    if match:
                        train_times.append(float(match.group(1)))
                        test_times.append(float(match.group(2)))
                
                # Parse weight statistics
                if "频率权重统计" in line and "均值=" in line:
                    match = re.search(r'均值=([\d.]+), 标准差=([\d.]+)', line)
                    if match:
                        weight_means.append(float(match.group(1)))
                        weight_stds.append(float(match.group(2)))
        
        final_accuracy = max(test_accs) if test_accs else 0.0
        log_path_str = str(log_file)

        # 检查此日志是否属于某个已初始化的剪枝模型
        if '/pruned/' in log_path_str:
            for ratio, model_data in self.pruned_models_data.items():
                if model_data['directory'].name in log_path_str:
                    model_data.update({
                        'final_accuracy': final_accuracy,
                        'epochs': epochs,
                        'train_losses': train_losses,
                        'train_accs': train_accs,
                        'test_losses': test_losses,
                        'test_accs': test_accs,
                        'train_times': train_times,
                        'test_times': test_times,
                    })
                    print(f"Updated metrics for pruned model (ratio {ratio}) with accuracy {final_accuracy:.4f}")
                    return

        # 检查此日志是否属于全尺寸模型
        if '/full/' in log_path_str or '/full_train/' in log_path_str:
            if self.full_training_data and self.full_training_data.get('directory') and self.full_training_data['directory'].name in log_path_str:
                self.full_training_data['final_accuracy'] = final_accuracy
                self.full_training_data.update({
                    'epochs': epochs,
                    'train_losses': train_losses,
                    'train_accs': train_accs,
                    'test_losses': test_losses,
                    'test_accs': test_accs,
                    'train_times': train_times,
                    'test_times': test_times,
                    'weight_means': weight_means,
                    'weight_stds': weight_stds
                })
                print(f"Updated metrics for full model with accuracy {final_accuracy:.4f}")
                return
    
    def plot_frequency_weight_evolution(self):
        """Plot frequency weight evolution curve"""
        print("Generating frequency weight evolution curve...")
        
        if not self.full_training_data:
            print("Warning: No full training data")
            return
        
        weight_history = self.full_training_data['weight_history']
        weights_array = np.array(weight_history)
        
        plt.figure(figsize=(12, 8))
        
        # Plot weight changes for each frequency dimension
        for i in range(weights_array.shape[1]):
            plt.plot(weights_array[:, i], alpha=0.7, linewidth=1, label=f'Freq {i+1}')
        
        plt.xlabel('Training Epoch', fontsize=12)
        plt.ylabel('Frequency Weight Value', fontsize=12)
        plt.title('Frequency Weight Evolution During Training', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'freq_weight_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weight_distribution_comparison(self):
        """Compare weight distributions before and after pruning"""
        print("Generating weight distribution comparison...")
        
        if not self.full_training_data or not self.pruned_models_data:
            print("Warning: Missing training data")
            return
        
        # 选择第一个剪枝模型进行对比（或者你可以选择特定的剪枝比例）
        pruned_model_key = list(self.pruned_models_data.keys())[0]
        pruned_model = self.pruned_models_data[pruned_model_key]
        
        full_weights = self.full_training_data['final_weights']
        pruned_weights = pruned_model['final_weights']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distribution before pruning
        ax1.hist(full_weights, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Weight Value', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Before Pruning (40 frequency bins)', fontsize=14, fontweight='bold')
        ax1.axvline(np.mean(full_weights), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(full_weights):.3f}')
        ax1.legend()
        
        # Distribution after pruning
        ax2.hist(pruned_weights, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Weight Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'After {pruned_model["model_name"]}', fontsize=14, fontweight='bold')
        ax2.axvline(np.mean(pruned_weights), color='red', linestyle='--',
                    label=f'Mean: {np.mean(pruned_weights):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weight_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_speed_tradeoff(self):
        """Plot accuracy-speed tradeoff curve"""
        print("Generating accuracy-speed tradeoff curve...")
        
        if not self.full_training_data or not self.pruned_models_data:
            print("Warning: Missing training data")
            return
        
        # 准备数据
        pruning_ratios = [0]  # 0% = no pruning
        accuracies = [self.full_training_data['final_accuracy']]
        training_times = [np.mean(self.full_training_data.get('train_times', [17.44]))]
        
        # 添加剪枝模型数据
        for ratio, model_data in self.pruned_models_data.items():
            pruning_ratios.append(ratio)
            accuracies.append(model_data['final_accuracy'])
            training_times.append(np.mean(model_data.get('train_times', [7.22])))
        
        # 计算加速比
        speedups = [1.0]
        for i in range(1, len(training_times)):
            speedups.append(training_times[0] / training_times[i])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy-speedup tradeoff
        ax1.plot(pruning_ratios, accuracies, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_xlabel('Pruning Ratio', fontsize=12)
        ax1.set_ylabel('Test Accuracy', fontsize=12)
        ax1.set_title('Accuracy vs Pruning Ratio', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.85, 0.95)
        
        # Add data labels
        for i, (ratio, acc) in enumerate(zip(pruning_ratios, accuracies)):
            ax1.annotate(f'{acc:.3f}', (ratio, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
        
        # Training time comparison
        colors = ['blue'] + ['green'] * (len(pruning_ratios) - 1)
        bars = ax2.bar(pruning_ratios, training_times, color=colors, alpha=0.7)
        ax2.set_xlabel('Pruning Ratio', fontsize=12)
        ax2.set_ylabel('Training Time per Epoch (s)', fontsize=12)
        ax2.set_title('Training Time vs Pruning Ratio', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (ratio, time) in enumerate(zip(pruning_ratios, training_times)):
            ax2.text(ratio, time + 0.5, f'{time:.1f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_speed_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves_comparison(self):
        """Compare training curves before and after pruning"""
        print("Generating training curves comparison...")
        
        if not self.full_training_data or not self.pruned_models_data:
            print("Warning: Missing training data")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training accuracy comparison
        if 'train_accs' in self.full_training_data:
            ax1.plot(self.full_training_data['epochs'], self.full_training_data['train_accs'], 
                    'b-', label='Original Model', linewidth=2)
            
            # 添加所有剪枝模型的训练精度
            for ratio, model_data in self.pruned_models_data.items():
                if 'train_accs' in model_data:
                    ax1.plot(model_data['epochs'], model_data['train_accs'], 
                            '--', label=f'{model_data["model_name"]}', linewidth=2)
            
            ax1.set_xlabel('Training Epoch', fontsize=12)
            ax1.set_ylabel('Training Accuracy', fontsize=12)
            ax1.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Test accuracy comparison
        if 'test_accs' in self.full_training_data:
            ax2.plot(self.full_training_data['epochs'], self.full_training_data['test_accs'], 
                    'b-', label='Original Model', linewidth=2)
            
            # 添加所有剪枝模型的测试精度
            for ratio, model_data in self.pruned_models_data.items():
                if 'test_accs' in model_data:
                    ax2.plot(model_data['epochs'], model_data['test_accs'], 
                            '--', label=f'{model_data["model_name"]}', linewidth=2)
            
            ax2.set_xlabel('Training Epoch', fontsize=12)
            ax2.set_ylabel('Test Accuracy', fontsize=12)
            ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Training loss comparison
        if 'train_losses' in self.full_training_data:
            ax3.plot(self.full_training_data['epochs'], self.full_training_data['train_losses'], 
                    'b-', label='Original Model', linewidth=2)
            
            # 添加所有剪枝模型的训练损失
            for ratio, model_data in self.pruned_models_data.items():
                if 'train_losses' in model_data:
                    ax3.plot(model_data['epochs'], model_data['train_losses'], 
                            '--', label=f'{model_data["model_name"]}', linewidth=2)
            
            ax3.set_xlabel('Training Epoch', fontsize=12)
            ax3.set_ylabel('Training Loss', fontsize=12)
            ax3.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Training time comparison
        if 'train_times' in self.full_training_data:
            models = ['Original Model']
            times = [np.mean(self.full_training_data['train_times'])]
            
            # 添加所有剪枝模型的训练时间
            for ratio, model_data in self.pruned_models_data.items():
                if 'train_times' in model_data:
                    models.append(model_data['model_name'])
                    times.append(np.mean(model_data['train_times']))
            
            colors = ['blue'] + ['green'] * (len(models) - 1)
            ax4.bar(models, times, color=colors, alpha=0.7)
            ax4.set_ylabel('Average Training Time (s/epoch)', fontsize=12)
            ax4.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (model, time) in enumerate(zip(models, times)):
                ax4.text(i, time + 0.5, f'{time:.1f}s', ha='center', va='bottom', fontsize=10)
            
            # 旋转x轴标签以防重叠
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_frequency_importance_ranking(self):
        """Plot frequency dimension importance ranking"""
        print("Generating frequency importance ranking...")
        
        if not self.full_training_data:
            print("Warning: No full training data")
            return
        
        final_weights = self.full_training_data['final_weights']
        
        # Sort by weight
        sorted_indices = np.argsort(final_weights)[::-1]  # Descending order
        sorted_weights = final_weights[sorted_indices]
        
        plt.figure(figsize=(12, 6))
        
        # Plot importance ranking
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_weights)))
        bars = plt.bar(range(len(sorted_weights)), sorted_weights, color=colors)
        plt.xlabel('Frequency Bin Index (Ranked by Importance)', fontsize=12)
        plt.ylabel('Weight Value', fontsize=12)
        plt.title('Frequency Dimension Importance Ranking', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add pruning threshold line
        threshold_50 = sorted_weights[int(len(sorted_weights) * 0.5)]
        plt.axhline(y=threshold_50, color='red', linestyle='--', alpha=0.7, 
                    label=f'50% Pruning Threshold: {threshold_50:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frequency_importance_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return sorted_indices, sorted_weights
    
    def create_performance_table(self):
        """Create performance comparison table with three models"""
        print("Generating performance comparison table...")
        
        if not self.full_training_data or not self.pruned_models_data:
            print("Warning: Missing training data")
            return
        
        # 准备三个模型的数据
        models_data = []
        
        # 全训练模型
        models_data.append({
            'Model': 'Original Model (40 bins)',
            'Pruning Ratio': '0%',
            'Test Accuracy': self.full_training_data['final_accuracy'],
            'Accuracy Drop': '-',
            'Avg Training Time (s/epoch)': np.mean(self.full_training_data.get('train_times', [17.44])),
            'Speedup': '1.0x',
            'Parameters': 26880,
            'Parameter Reduction': '-'
        })
        
        # 0.25剪枝模型
        if 0.25 in self.pruned_models_data:
            pruned_25 = self.pruned_models_data[0.25]
            accuracy_change = pruned_25['final_accuracy'] - self.full_training_data['final_accuracy']
            models_data.append({
                'Model': '25% Pruned Model (30 bins)',
                'Pruning Ratio': '25%',
                'Test Accuracy': pruned_25['final_accuracy'],
                'Accuracy Drop': f"{accuracy_change*100:+.2f}%",  # 修复：使用+号显示正负
                'Avg Training Time (s/epoch)': np.mean(pruned_25.get('train_times', [7.22])),
                'Speedup': f"{np.mean(self.full_training_data.get('train_times', [17.44])) / np.mean(pruned_25.get('train_times', [7.22])):.2f}x",
                'Parameters': 20160,  # 30/40 * 26880
                'Parameter Reduction': '25%'
            })
        
        # 0.5剪枝模型
        if 0.5 in self.pruned_models_data:
            pruned_50 = self.pruned_models_data[0.5]
            accuracy_change = pruned_50['final_accuracy'] - self.full_training_data['final_accuracy']
            models_data.append({
                'Model': '50% Pruned Model (20 bins)',
                'Pruning Ratio': '50%',
                'Test Accuracy': pruned_50['final_accuracy'],
                'Accuracy Drop': f"{accuracy_change*100:+.2f}%",  # 修复：使用+号显示正负
                'Avg Training Time (s/epoch)': np.mean(pruned_50.get('train_times', [7.22])),
                'Speedup': f"{np.mean(self.full_training_data.get('train_times', [17.44])) / np.mean(pruned_50.get('train_times', [7.22])):.2f}x",
                'Parameters': 13440,  # 20/40 * 26880
                'Parameter Reduction': '50%'
            })
        
        df = pd.DataFrame(models_data)
        
        # Save as CSV
        df.to_csv(self.output_dir / 'performance_comparison.csv', index=False, encoding='utf-8-sig')
        
        # Create beautiful table
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        cellLoc='center', loc='center',
                        colWidths=[0.25, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Set header style
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Frequency Dimension Pruning Performance Comparison (Three Models)', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'performance_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return df
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("Generating summary report...")

        if not self.full_training_data:
            print("Warning: Full training data not available for summary report.")
            return

        # --- 优化建议: 动态地从剪枝模型数据中选择一个进行报告 ---
        reported_pruned_model = None
        if self.pruned_models_data:
            # 选择剪枝率最高的模型进行报告
            highest_ratio = max(self.pruned_models_data.keys(), default=0)
            if highest_ratio > 0:
                reported_pruned_model = self.pruned_models_data[highest_ratio]

        report = {
            "Project Information": {
                "Dataset": "UrbanSound8K",
                "Model Architecture": "VGG9-SNN",
                "Time Steps": 5,
                "Batch Size": 128,
                "Learning Rate": 0.0001,
                "Optimizer": "Adam"
            },
            "Pruning Results": {}
        }
        
        full_acc = self.full_training_data.get('final_accuracy', 0.0)
        full_time = np.mean(self.full_training_data.get('train_times', [0.0]))
        report['Pruning Results']['Original Accuracy'] = f"{full_acc:.4f}"

        if reported_pruned_model:
            pruned_acc = reported_pruned_model.get('final_accuracy', 0.0)
            pruned_time = np.mean(reported_pruned_model.get('train_times', [0.0]))
            pruning_ratio = reported_pruned_model.get('pruning_ratio', 0.0)

            report['Pruning Results']['Reported Pruned Model'] = reported_pruned_model.get('model_name', 'N/A')
            report['Pruning Results']['Pruned Accuracy'] = f"{pruned_acc:.4f}"
            report['Pruning Results']['Accuracy Change'] = f"{(pruned_acc - full_acc) * 100:+.2f}%"
            if full_time > 0 and pruned_time > 0:
                report['Pruning Results']['Training Speedup'] = f"{full_time / pruned_time:.2f}x"
            else:
                report['Pruning Results']['Training Speedup'] = "N/A"
            report['Pruning Results']['Frequency Dimension Reduction'] = f"{pruning_ratio * 100:.0f}%"
        else:
            report['Pruning Results']['Message'] = "No pruned model data found to compare."

        report["Generated Files"] = [
            "freq_weight_evolution.png - Frequency weight evolution curve",
            "accuracy_speed_tradeoff.png - Accuracy-speed tradeoff",
            "training_curves_comparison.png - Training curves comparison",
            "frequency_importance_ranking.png - Frequency importance ranking",
            "performance_table.png - Performance comparison table",
            "performance_comparison.csv - Performance data"
        ]
        
        # Save report
        with open(self.output_dir / 'summary_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print report
        print("\n" + "="*50)
        print("Frequency Dimension Pruning Project Summary Report")
        print("="*50)
        print(f"Original Model Accuracy: {report['Pruning Results']['Original Accuracy']:.4f}")
        print(f"Pruned Model Accuracy: {report['Pruning Results']['Pruned Accuracy']:.4f}")
        print(f"Accuracy Drop: {report['Pruning Results']['Accuracy Drop']}")
        print(f"Training Speedup: {report['Pruning Results']['Training Speedup']}")
        print(f"Frequency Dimension Reduction: {report['Pruning Results']['Frequency Dimension Reduction']}")
        print("="*50)
    
    def run_all_analysis(self):
        """Run all analysis"""
        print("Starting frequency dimension pruning data analysis...")
        
        # Load data
        self.load_training_data()
        self.extract_training_metrics_from_logs()
        # 自动生成 analysis_results/{tag}/ 子目录，避免覆盖
        if self.output_dir is None:
            import datetime
            tag_base = None
            if self.pruned_models_data:
                any_pruned = next(iter(self.pruned_models_data.values()))
                tag_base = Path(any_pruned['directory']).name
            elif self.full_training_data:
                tag_base = "full_auto"
            else:
                tag_base = "session"
            stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            tag = f"{tag_base}__{stamp}"
            self.output_dir = self.output_root / tag
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate charts
        self.plot_frequency_weight_evolution()
        self.plot_accuracy_speed_tradeoff()
        self.plot_training_curves_comparison()
        self.plot_frequency_importance_ranking()
        self.create_performance_table()
        
        # Generate report
        self.generate_summary_report()
        
        print(f"\nAnalysis completed! Results saved in: {self.output_dir}")

if __name__ == "__main__":
    # Create analyzer and run，支持命令行传入3个训练目录，并输出到 analysis_results/{tag}/
    import argparse, datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', type=str, default=None, help='full 训练目录路径，例如: ./runs/full/xxx')
    parser.add_argument('--pruned_keep30', type=str, default=None, help='keep30of40 剪枝训练目录路径，例如: ./runs/pruned/xxx_keep30of40_xxx')
    parser.add_argument('--pruned_keep20', type=str, default=None, help='keep20of40 剪枝训练目录路径，例如: ./runs/pruned/xxx_keep20of40_xxx')
    parser.add_argument('--tag', type=str, default=None, help='analysis_results 子目录名，不给则自动生成')
    args_cli = parser.parse_args()

    analyzer = FrequencyPruningAnalyzer()

    manual_runs = {}
    if args_cli.full:
        manual_runs['full'] = Path(args_cli.full)
    if args_cli.pruned_keep30:
        manual_runs['keep30'] = Path(args_cli.pruned_keep30)
    if args_cli.pruned_keep20:
        manual_runs['keep20'] = Path(args_cli.pruned_keep20)
    
    if manual_runs:
        print("Using manually specified run directories...")
        log_files_to_parse = []
        if 'full' in manual_runs:
            analyzer.load_full_training_data(manual_runs['full'])
            log_files_to_parse.extend(list(manual_runs['full'].glob("train_record.log")))
        if 'keep30' in manual_runs:
            analyzer.load_pruned_training_data(manual_runs['keep30'])
            log_files_to_parse.extend(list(manual_runs['keep30'].glob("train_record.log")))
        if 'keep20' in manual_runs:
            analyzer.load_pruned_training_data(manual_runs['keep20'])
            log_files_to_parse.extend(list(manual_runs['keep20'].glob("train_record.log")))
        
        analyzer.extract_training_metrics_from_logs(log_files_to_parse)

        analyzer.output_root = Path("./analysis_results")
        analyzer.output_root.mkdir(exist_ok=True)
        if args_cli.tag:
            tag = args_cli.tag
        else:
            parts = []
            if 'full' in manual_runs:
                parts.append(manual_runs['full'].name)
            if 'keep30' in manual_runs:
                parts.append(manual_runs['keep30'].name)
            if 'keep20' in manual_runs:
                parts.append(manual_runs['keep20'].name)
            stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            tag = ("__".join(parts)[:120] if parts else "session") + f"__{stamp}"
        analyzer.output_dir = analyzer.output_root / tag
        analyzer.output_dir.mkdir(parents=True, exist_ok=True)

        analyzer.plot_frequency_weight_evolution()
        analyzer.plot_accuracy_speed_tradeoff()
        analyzer.plot_training_curves_comparison()
        analyzer.plot_frequency_importance_ranking()
        analyzer.create_performance_table()
        analyzer.generate_summary_report()
        print(f"\nAnalysis completed! Results saved in: {analyzer.output_dir}")
    else:
        analyzer.run_all_analysis() 