# #!/usr/bin/env python3
# """
# Standalone visualization generator for PCM-LLM analysis results.
# This script reads the latest comprehensive data CSV and generates multiple specialized PNG charts.
# """

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import glob

# def create_specialized_visualizations():
#     """Create specialized visualization charts from the latest analysis data."""
    
#     # Find the latest comprehensive data CSV
#     output_path = Path("analysis_output")
#     data_files = list(output_path.glob("ultra_comprehensive_data_*.csv"))
    
#     if not data_files:
#         print("❌ No comprehensive data files found!")
#         return
    
#     # Get the latest file
#     latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
#     print(f"📊 Loading data from: {latest_file}")
    
#     # Load the data
#     df = pd.read_csv(latest_file)
#     print(f"📈 Loaded {len(df)} samples")
    
#     # Generate timestamp for output files
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     print("\n🎨 Creating specialized visualizations...")
    
#     # 1. Method Performance Comparison Chart
#     plt.figure(figsize=(12, 8))
#     method_stats = df.groupby('method').agg({
#         'score_preservation': ['mean', 'std'],
#         'compressed_accuracy': ['mean', 'std'],
#         'cost_benefit_ratio': ['mean', 'std']
#     })
    
#     methods = method_stats.index
#     preservation_means = method_stats[('score_preservation', 'mean')]
#     preservation_stds = method_stats[('score_preservation', 'std')].fillna(0)
    
#     x = range(len(methods))
#     bars = plt.bar(x, preservation_means, yerr=preservation_stds, capsize=5, alpha=0.8, 
#             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
#     plt.xlabel('Compression Method', fontsize=12)
#     plt.ylabel('Score Preservation (%)', fontsize=12)
#     plt.title('Method Performance Comparison with Error Bars', fontsize=14, fontweight='bold')
#     plt.xticks(x, [m.replace(' ', '\n') for m in methods])
#     plt.grid(True, alpha=0.3)
    
#     # Add value labels on bars
#     for bar, mean_val in zip(bars, preservation_means):
#         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
#                 f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
#     plt.tight_layout()
#     plt.savefig(output_path / f"method_performance_{timestamp}.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("   ✅ Method performance chart created")
    
#     # 2. Compression Ratio Distribution Analysis
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 3, 1)
#     plt.hist(df['actual_compression_ratio'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
#     plt.xlabel('Actual Compression Ratio')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Achieved Ratios')
#     plt.grid(True, alpha=0.3)
    
#     plt.subplot(1, 3, 2)
#     colors = ['#FF9999', '#66B2FF', '#99FF99']
#     for i, task in enumerate(df['task'].unique()):
#         task_data = df[df['task'] == task]['actual_compression_ratio']
#         plt.hist(task_data, alpha=0.6, label=task, bins=20, color=colors[i % len(colors)])
#     plt.xlabel('Actual Compression Ratio')
#     plt.ylabel('Frequency')
#     plt.title('Ratio Distribution by Task')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.subplot(1, 3, 3)
#     # Create box plot manually since pandas boxplot might have issues
#     method_ratios = [df[df['method'] == method]['actual_compression_ratio'].values 
#                     for method in df['method'].unique()]
#     plt.boxplot(method_ratios, labels=df['method'].unique())
#     plt.title('Compression Ratio by Method')
#     plt.ylabel('Actual Compression Ratio')
#     plt.xticks(rotation=45)
    
#     plt.tight_layout()
#     plt.savefig(output_path / f"ratio_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("   ✅ Ratio distribution charts created")
    
#     # 3. Performance vs Efficiency Analysis
#     plt.figure(figsize=(12, 10))
    
#     plt.subplot(2, 2, 1)
#     scatter1 = plt.scatter(df['actual_compression_ratio'], df['score_preservation'], 
#                           c=df['cost_benefit_ratio'], cmap='viridis', alpha=0.7, s=50)
#     plt.xlabel('Actual Compression Ratio')
#     plt.ylabel('Score Preservation (%)')
#     plt.title('Performance vs Compression Ratio')
#     plt.colorbar(scatter1, label='Cost-Benefit Ratio')
#     plt.grid(True, alpha=0.3)
    
#     plt.subplot(2, 2, 2)
#     method_colors = {'Llmlingua2': 'red', 'Selective Context': 'blue', 'Naive Truncation': 'green'}
#     colors = [method_colors.get(m, 'gray') for m in df['method']]
#     plt.scatter(df['compressed_accuracy'], df['score_preservation'], c=colors, alpha=0.7, s=50)
#     plt.xlabel('Compressed Accuracy (%)')
#     plt.ylabel('Score Preservation (%)')
#     plt.title('Accuracy vs Preservation by Method')
#     plt.grid(True, alpha=0.3)
    
#     # Add legend
#     for method, color in method_colors.items():
#         if method in df['method'].values:
#             plt.scatter([], [], c=color, label=method, s=50)
#     plt.legend()
    
#     plt.subplot(2, 2, 3)
#     task_perf = df.groupby('task')['score_preservation'].mean().sort_values(ascending=True)
#     bars = plt.barh(range(len(task_perf)), task_perf.values, color='lightcoral', alpha=0.8)
#     plt.yticks(range(len(task_perf)), task_perf.index)
#     plt.xlabel('Average Score Preservation (%)')
#     plt.title('Task Performance Ranking')
#     plt.grid(True, alpha=0.3)
    
#     # Add value labels
#     for i, (bar, val) in enumerate(zip(bars, task_perf.values)):
#         plt.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold')
    
#     plt.subplot(2, 2, 4)
#     efficiency_data = df['compression_efficiency'].dropna()
#     if not efficiency_data.empty:
#         plt.hist(efficiency_data, bins=25, alpha=0.8, color='gold', edgecolor='black')
#         plt.xlabel('Compression Efficiency (%)')
#         plt.ylabel('Frequency')
#         plt.title('Efficiency Distribution')
#         plt.grid(True, alpha=0.3)
#     else:
#         plt.text(0.5, 0.5, 'No Efficiency Data Available', ha='center', va='center', 
#                 transform=plt.gca().transAxes, fontsize=12)
    
#     plt.tight_layout()
#     plt.savefig(output_path / f"performance_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("   ✅ Performance analysis charts created")
    
#     # 4. Correlation Matrix
#     plt.figure(figsize=(12, 10))
#     numeric_cols = ['actual_compression_ratio', 'score_preservation', 'compressed_accuracy', 
#                    'accuracy_drop', 'cost_benefit_ratio']
    
#     # Add compression_efficiency if it has valid data
#     if not df['compression_efficiency'].isna().all():
#         numeric_cols.append('compression_efficiency')
    
#     correlation_data = df[numeric_cols].select_dtypes(include=[np.number])
#     correlation_matrix = correlation_data.corr()
    
#     # Create heatmap
#     mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
#     sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
#                square=True, fmt='.3f', mask=mask, cbar_kws={"shrink": .8})
#     plt.title('Correlation Matrix of Performance Metrics', fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(output_path / f"correlation_matrix_{timestamp}.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("   ✅ Correlation matrix created")
    
#     # 5. Summary Dashboard
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#     fig.suptitle('PCM-LLM Analysis Summary Dashboard', fontsize=16, fontweight='bold')
    
#     # Overall stats
#     ax1 = axes[0, 0]
#     summary_stats = {
#         'Total Samples': len(df),
#         'Unique Ratios': df['actual_compression_ratio'].nunique(),
#         'Methods': df['method'].nunique(),
#         'Tasks': df['task'].nunique()
#     }
#     bars1 = ax1.bar(summary_stats.keys(), summary_stats.values(), 
#                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
#     ax1.set_title('Dataset Overview')
#     ax1.set_ylabel('Count')
#     for bar, val in zip(bars1, summary_stats.values()):
#         ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(summary_stats.values())*0.02, 
#                 str(val), ha='center', va='bottom', fontweight='bold')
    
#     # Method performance comparison
#     ax2 = axes[0, 1]
#     method_avg = df.groupby('method')['score_preservation'].mean().sort_values(ascending=False)
#     bars2 = ax2.bar(range(len(method_avg)), method_avg.values, 
#                    color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
#     ax2.set_title('Average Score Preservation by Method')
#     ax2.set_ylabel('Score Preservation (%)')
#     ax2.set_xticks(range(len(method_avg)))
#     ax2.set_xticklabels([m.replace(' ', '\n') for m in method_avg.index])
#     for bar, val in zip(bars2, method_avg.values):
#         ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
#                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
#     # Task performance comparison
#     ax3 = axes[0, 2]
#     task_avg = df.groupby('task')['score_preservation'].mean().sort_values(ascending=False)
#     bars3 = ax3.bar(range(len(task_avg)), task_avg.values, 
#                    color=['#96CEB4', '#FECA57', '#FF9FF3'])
#     ax3.set_title('Average Score Preservation by Task')
#     ax3.set_ylabel('Score Preservation (%)')
#     ax3.set_xticks(range(len(task_avg)))
#     ax3.set_xticklabels(task_avg.index)
#     for bar, val in zip(bars3, task_avg.values):
#         ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
#                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
#     # Distribution of compression ratios
#     ax4 = axes[1, 0]
#     ax4.hist(df['actual_compression_ratio'], bins=25, alpha=0.8, color='lightblue', edgecolor='black')
#     ax4.set_title('Compression Ratio Distribution')
#     ax4.set_xlabel('Actual Compression Ratio')
#     ax4.set_ylabel('Frequency')
#     ax4.grid(True, alpha=0.3)
    
#     # Score preservation distribution
#     ax5 = axes[1, 1]
#     ax5.hist(df['score_preservation'], bins=25, alpha=0.8, color='lightgreen', edgecolor='black')
#     ax5.set_title('Score Preservation Distribution')
#     ax5.set_xlabel('Score Preservation (%)')
#     ax5.set_ylabel('Frequency')
#     ax5.grid(True, alpha=0.3)
    
#     # Cost-benefit analysis
#     ax6 = axes[1, 2]
#     ax6.scatter(df['actual_compression_ratio'], df['cost_benefit_ratio'], 
#                alpha=0.6, c='orange', s=30)
#     ax6.set_title('Cost-Benefit vs Compression Ratio')
#     ax6.set_xlabel('Actual Compression Ratio')
#     ax6.set_ylabel('Cost-Benefit Ratio')
#     ax6.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(output_path / f"summary_dashboard_{timestamp}.png", dpi=300, bbox_inches='tight')
#     plt.close()
#     print("   ✅ Summary dashboard created")
    
#     print(f"\n🎉 All visualizations created successfully!")
#     print(f"📊 Generated files with timestamp: {timestamp}")
#     print(f"   • Method performance: method_performance_{timestamp}.png")
#     print(f"   • Ratio distribution: ratio_distribution_{timestamp}.png") 
#     print(f"   • Performance analysis: performance_analysis_{timestamp}.png")
#     print(f"   • Correlation matrix: correlation_matrix_{timestamp}.png")
#     print(f"   • Summary dashboard: summary_dashboard_{timestamp}.png")

# if __name__ == "__main__":
#     create_specialized_visualizations()
