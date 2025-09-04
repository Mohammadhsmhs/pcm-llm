#!/usr/bin/env python3
"""
Visualization Module for PCM-LLM Analysis
Handles creation of comprehensive visualizations and charts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Optional imports for visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available. Visualizations will be skipped.")

class VisualizationGenerator:
    """Generates comprehensive visualizations for analysis results."""
    
    def __init__(self, output_dir: str = "analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.available = VISUALIZATION_AVAILABLE

    def create_all_visualizations(self, df: pd.DataFrame, timestamp: str) -> List[str]:
        """Create all visualization charts."""
        if not self.available:
            logger.warning("Visualization libraries not available. Skipping chart generation.")
            return []
        
        generated_files = []
        
        try:
            # Main comprehensive dashboard
            main_viz = self._create_comprehensive_dashboard(df, timestamp)
            if main_viz:
                generated_files.append(main_viz)
            
            # Specialized visualizations
            specialized_files = self._create_specialized_visualizations(df, timestamp)
            generated_files.extend(specialized_files)
            
            logger.info(f"Generated {len(generated_files)} visualization files")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return generated_files

    def _create_comprehensive_dashboard(self, df: pd.DataFrame, timestamp: str) -> Optional[str]:
        """Create main comprehensive visualization dashboard."""
        try:
            viz_file = self.output_dir / f"comprehensive_dashboard_{timestamp}.png"
            
            # Create large figure with multiple subplots
            fig, axes = plt.subplots(3, 4, figsize=(24, 18))
            fig.suptitle('PCM-LLM Comprehensive Analysis Dashboard', fontsize=20, fontweight='bold')
            
            # 1. Score Preservation by Compression Ratio
            self._plot_preservation_by_ratio(df, axes[0, 0])
            
            # 2. Method Performance Comparison
            self._plot_method_comparison(df, axes[0, 1])
            
            # 3. Task Performance Heatmap
            self._plot_task_method_heatmap(df, axes[0, 2])
            
            # 4. Cost-Benefit vs Performance Scatter
            self._plot_cost_benefit_scatter(df, axes[0, 3])
            
            # 5. Compression Efficiency Distribution
            self._plot_efficiency_distribution(df, axes[1, 0])
            
            # 6. Accuracy Distribution by Method
            self._plot_accuracy_distribution(df, axes[1, 1])
            
            # 7. Performance Trends
            self._plot_performance_trends(df, axes[1, 2])
            
            # 8. Ratio Achievement Analysis
            self._plot_ratio_achievement(df, axes[1, 3])
            
            # 9. Model Performance (if multiple models)
            self._plot_model_performance(df, axes[2, 0])
            
            # 10. Outlier Detection
            self._plot_outlier_detection(df, axes[2, 1])
            
            # 11. Correlation Matrix
            self._plot_correlation_matrix(df, axes[2, 2])
            
            # 12. Summary Statistics
            self._plot_summary_statistics(df, axes[2, 3])
            
            plt.tight_layout()
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comprehensive dashboard saved: {viz_file}")
            return str(viz_file)
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard: {e}")
            return None

    def _create_specialized_visualizations(self, df: pd.DataFrame, timestamp: str) -> List[str]:
        """Create specialized visualization charts."""
        files = []
        
        try:
            # Method performance chart
            method_file = self._create_method_performance_chart(df, timestamp)
            if method_file:
                files.append(method_file)
            
            # Ratio distribution chart  
            ratio_file = self._create_ratio_distribution_chart(df, timestamp)
            if ratio_file:
                files.append(ratio_file)
            
            # Performance analysis chart
            performance_file = self._create_performance_analysis_chart(df, timestamp)
            if performance_file:
                files.append(performance_file)
            
            # Correlation matrix
            correlation_file = self._create_correlation_chart(df, timestamp)
            if correlation_file:
                files.append(correlation_file)
            
            # Method-Task-Ratio analysis chart (NEW!)
            method_task_ratio_file = self._create_method_task_ratio_chart(df, timestamp)
            if method_task_ratio_file:
                files.append(method_task_ratio_file)
                
        except Exception as e:
            logger.error(f"Error creating specialized visualizations: {e}")
        
        return files

    def _plot_preservation_by_ratio(self, df: pd.DataFrame, ax):
        """Plot score preservation by compression ratio."""
        try:
            if 'actual_compression_ratio' in df.columns and 'score_preservation' in df.columns:
                # Create ratio bins for cleaner visualization
                df['ratio_bin'] = pd.cut(df['actual_compression_ratio'], bins=10, labels=False)
                ratio_stats = df.groupby('ratio_bin')['score_preservation'].agg(['mean', 'std']).fillna(0)
                
                if not ratio_stats.empty:
                    ratio_stats['mean'].plot(kind='bar', ax=ax, yerr=ratio_stats['std'], capsize=3)
                    ax.set_title('Score Preservation by Compression Ratio')
                    ax.set_ylabel('Score Preservation (%)')
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, 'No ratio data available', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Missing required columns', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_method_comparison(self, df: pd.DataFrame, ax):
        """Plot method performance comparison."""
        try:
            if 'method' in df.columns and 'score_preservation' in df.columns:
                method_perf = df.groupby('method')['score_preservation'].mean().sort_values(ascending=False)
                method_perf.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Method Performance Comparison')
                ax.set_ylabel('Avg Score Preservation (%)')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'Missing method data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_task_method_heatmap(self, df: pd.DataFrame, ax):
        """Plot task vs method performance heatmap."""
        try:
            if 'task' in df.columns and 'method' in df.columns and 'score_preservation' in df.columns:
                pivot = df.pivot_table(values='score_preservation', index='method', columns='task', aggfunc='mean')
                if not pivot.empty:
                    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax)
                    ax.set_title('Performance Heatmap (Method vs Task)')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Missing required data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_cost_benefit_scatter(self, df: pd.DataFrame, ax):
        """Plot cost-benefit vs performance scatter."""
        try:
            if 'cost_benefit_ratio' in df.columns and 'score_preservation' in df.columns:
                scatter = ax.scatter(df['cost_benefit_ratio'], df['score_preservation'], 
                                   c=df['actual_compression_ratio'] if 'actual_compression_ratio' in df.columns else 'blue',
                                   cmap='viridis', alpha=0.6)
                ax.set_xlabel('Cost-Benefit Ratio')
                ax.set_ylabel('Score Preservation (%)')
                ax.set_title('Cost-Benefit vs Performance')
                if 'actual_compression_ratio' in df.columns:
                    plt.colorbar(scatter, ax=ax, label='Compression Ratio')
            else:
                ax.text(0.5, 0.5, 'Missing cost-benefit data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_efficiency_distribution(self, df: pd.DataFrame, ax):
        """Plot compression efficiency distribution."""
        try:
            if 'compression_efficiency' in df.columns:
                efficiency_data = df['compression_efficiency'].dropna()
                if not efficiency_data.empty:
                    efficiency_data.hist(bins=20, ax=ax, alpha=0.7, color='orange', edgecolor='black')
                    ax.set_title('Compression Efficiency Distribution')
                    ax.set_xlabel('Compression Efficiency (%)')
                    ax.set_ylabel('Frequency')
                else:
                    ax.text(0.5, 0.5, 'No efficiency data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No efficiency column', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_accuracy_distribution(self, df: pd.DataFrame, ax):
        """Plot accuracy distribution by method."""
        try:
            if 'method' in df.columns and 'compressed_accuracy' in df.columns:
                methods = df['method'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
                
                for method, color in zip(methods, colors):
                    method_data = df[df['method'] == method]['compressed_accuracy'].dropna()
                    if not method_data.empty:
                        ax.hist(method_data, alpha=0.5, label=method, color=color, bins=15)
                
                ax.set_title('Accuracy Distribution by Method')
                ax.set_xlabel('Compressed Accuracy (%)')
                ax.set_ylabel('Frequency')
                if len(methods) > 0:
                    ax.legend()
            else:
                ax.text(0.5, 0.5, 'Missing data for accuracy distribution', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_performance_trends(self, df: pd.DataFrame, ax):
        """Plot performance trends."""
        try:
            if 'actual_compression_ratio' in df.columns and 'score_preservation' in df.columns:
                # Sample data for trend line if too many points
                sample_df = df.sample(n=min(500, len(df))) if len(df) > 500 else df
                
                ax.scatter(sample_df['actual_compression_ratio'], sample_df['score_preservation'], 
                          alpha=0.6, c='blue', s=20)
                ax.set_xlabel('Compression Ratio')
                ax.set_ylabel('Score Preservation (%)')
                ax.set_title('Performance Trends')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Missing trend data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_ratio_achievement(self, df: pd.DataFrame, ax):
        """Plot target vs actual compression ratios."""
        try:
            if 'target_ratio' in df.columns and 'actual_compression_ratio' in df.columns:
                # Sample for visualization
                sample_df = df.sample(n=min(200, len(df))) if len(df) > 200 else df
                
                ax.scatter(sample_df['target_ratio'], sample_df['actual_compression_ratio'], 
                          alpha=0.6, c='green', s=30)
                
                # Perfect match line
                min_val = min(sample_df['target_ratio'].min(), sample_df['actual_compression_ratio'].min())
                max_val = max(sample_df['target_ratio'].max(), sample_df['actual_compression_ratio'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Match')
                
                ax.set_xlabel('Target Ratio')
                ax.set_ylabel('Actual Ratio')
                ax.set_title('Target vs Actual Ratios')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'Missing ratio data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_model_performance(self, df: pd.DataFrame, ax):
        """Plot model performance comparison."""
        try:
            if 'model' in df.columns and 'score_preservation' in df.columns:
                models = df['model'].unique()
                if len(models) > 1:
                    model_perf = df.groupby('model')['score_preservation'].mean()
                    model_perf.plot(kind='bar', ax=ax, color='lightgreen')
                    ax.set_title('Performance by Model')
                    ax.set_ylabel('Avg Score Preservation (%)')
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, f'Single Model:\n{models[0] if models else "Unknown"}', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No model data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_outlier_detection(self, df: pd.DataFrame, ax):
        """Plot outlier detection visualization."""
        try:
            if 'score_preservation' in df.columns:
                data = df['score_preservation'].dropna()
                if len(data) > 0:
                    # Simple boxplot for outlier detection
                    bp = ax.boxplot(data, labels=['Score Preservation'])
                    ax.set_title('Outlier Detection')
                    ax.set_ylabel('Score Preservation (%)')
                else:
                    ax.text(0.5, 0.5, 'No data for outlier detection', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Missing preservation data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_correlation_matrix(self, df: pd.DataFrame, ax):
        """Plot correlation matrix."""
        try:
            numeric_cols = ['actual_compression_ratio', 'score_preservation', 'compressed_accuracy', 
                           'accuracy_drop', 'cost_benefit_ratio']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = df[available_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f', ax=ax)
                ax.set_title('Correlation Matrix')
            else:
                ax.text(0.5, 0.5, 'Insufficient numeric columns', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_summary_statistics(self, df: pd.DataFrame, ax):
        """Plot summary statistics."""
        try:
            summary_metrics = {}
            
            if 'score_preservation' in df.columns:
                mean_val = df['score_preservation'].mean()
                if not pd.isna(mean_val):
                    summary_metrics['Avg Preservation'] = float(mean_val)  # Convert to Python float
            
            if 'compressed_accuracy' in df.columns:
                mean_val = df['compressed_accuracy'].mean()
                if not pd.isna(mean_val):
                    summary_metrics['Avg Accuracy'] = float(mean_val)  # Convert to Python float
            
            if 'cost_benefit_ratio' in df.columns:
                mean_val = df['cost_benefit_ratio'].mean()
                if not pd.isna(mean_val):
                    summary_metrics['Avg Cost-Benefit'] = float(mean_val)  # Convert to Python float
            
            if 'compression_efficiency' in df.columns and not df['compression_efficiency'].isna().all():
                mean_val = df['compression_efficiency'].mean()
                if not pd.isna(mean_val):
                    summary_metrics['Avg Efficiency'] = float(mean_val)  # Convert to Python float
            
            if summary_metrics:
                # Create colors list - ensure they are valid matplotlib colors
                colors = ['gold', 'silver', '#CD7F32', 'lightblue']  # Using hex for bronze
                colors = colors[:len(summary_metrics)]
                
                # Convert to lists and ensure proper types
                keys = list(summary_metrics.keys())
                values = list(summary_metrics.values())
                
                # Create the bar chart with explicit parameters
                bars = ax.bar(keys, values, color=colors, alpha=0.8)
                ax.set_title('Summary Statistics')
                ax.set_ylabel('Score (%)')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels
                max_val = max(values) if values else 1
                for i, (bar, value) in enumerate(zip(bars, values)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.02, 
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No summary data available', ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            logger.error(f"Error in _plot_summary_statistics: {e}")
            # Fallback to a simple text display
            ax.clear()
            ax.text(0.5, 0.5, 'Summary Statistics\n(Chart Error)', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, fontweight='bold')
            ax.set_title('Summary Statistics')
            # Remove axes for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])

    def _create_method_performance_chart(self, df: pd.DataFrame, timestamp: str) -> Optional[str]:
        """Create detailed method performance chart."""
        try:
            if 'method' not in df.columns or 'score_preservation' not in df.columns:
                return None
            
            plt.figure(figsize=(12, 8))
            
            method_stats = df.groupby('method').agg({
                'score_preservation': ['mean', 'std'],
                'compressed_accuracy': ['mean', 'std']
            })
            
            methods = method_stats.index
            preservation_means = method_stats[('score_preservation', 'mean')]
            preservation_stds = method_stats[('score_preservation', 'std')].fillna(0)
            
            # Create colors list that matches the number of methods
            available_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            colors = [available_colors[i % len(available_colors)] for i in range(len(methods))]
            
            x = range(len(methods))
            bars = plt.bar(x, preservation_means, yerr=preservation_stds, capsize=5, 
                          alpha=0.8, color=colors)
            
            plt.xlabel('Compression Method', fontsize=12)
            plt.ylabel('Score Preservation (%)', fontsize=12)
            plt.title('Method Performance Comparison with Error Bars', fontsize=14, fontweight='bold')
            plt.xticks(x, methods, rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, mean_val in zip(bars, preservation_means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            method_file = self.output_dir / f"method_performance_{timestamp}.png"
            plt.savefig(method_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(method_file)
            
        except Exception as e:
            logger.error(f"Error creating method performance chart: {e}")
            return None

    def _create_ratio_distribution_chart(self, df: pd.DataFrame, timestamp: str) -> Optional[str]:
        """Create ratio distribution analysis chart."""
        try:
            if 'actual_compression_ratio' not in df.columns:
                return None
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Overall distribution
            axes[0].hist(df['actual_compression_ratio'], bins=30, alpha=0.7, 
                        color='skyblue', edgecolor='black')
            axes[0].set_title('Distribution of Compression Ratios')
            axes[0].set_xlabel('Actual Compression Ratio')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Distribution by task
            if 'task' in df.columns:
                colors = ['#FF9999', '#66B2FF', '#99FF99']
                for i, task in enumerate(df['task'].unique()):
                    task_data = df[df['task'] == task]['actual_compression_ratio']
                    axes[1].hist(task_data, alpha=0.6, label=task, bins=20, 
                               color=colors[i % len(colors)])
                axes[1].set_title('Distribution by Task')
                axes[1].set_xlabel('Actual Compression Ratio')
                axes[1].set_ylabel('Frequency')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'No task data', ha='center', va='center', 
                           transform=axes[1].transAxes)
            
            # Box plot by method
            if 'method' in df.columns:
                method_ratios = [df[df['method'] == method]['actual_compression_ratio'].values 
                               for method in df['method'].unique()]
                axes[2].boxplot(method_ratios, labels=df['method'].unique())
                axes[2].set_title('Ratio Distribution by Method')
                axes[2].set_ylabel('Actual Compression Ratio')
                axes[2].tick_params(axis='x', rotation=45)
            else:
                axes[2].text(0.5, 0.5, 'No method data', ha='center', va='center', 
                           transform=axes[2].transAxes)
            
            plt.tight_layout()
            ratio_file = self.output_dir / f"ratio_distribution_{timestamp}.png"
            plt.savefig(ratio_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(ratio_file)
            
        except Exception as e:
            logger.error(f"Error creating ratio distribution chart: {e}")
            return None

    def _create_performance_analysis_chart(self, df: pd.DataFrame, timestamp: str) -> Optional[str]:
        """Create performance analysis chart."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Performance vs Compression Ratio
            if 'actual_compression_ratio' in df.columns and 'score_preservation' in df.columns:
                scatter1 = axes[0, 0].scatter(df['actual_compression_ratio'], df['score_preservation'], 
                                            c=df['cost_benefit_ratio'] if 'cost_benefit_ratio' in df.columns else 'blue',
                                            cmap='viridis', alpha=0.7, s=50)
                axes[0, 0].set_xlabel('Compression Ratio')
                axes[0, 0].set_ylabel('Score Preservation (%)')
                axes[0, 0].set_title('Performance vs Compression')
                axes[0, 0].grid(True, alpha=0.3)
                
                if 'cost_benefit_ratio' in df.columns:
                    plt.colorbar(scatter1, ax=axes[0, 0], label='Cost-Benefit')
            
            # Accuracy vs Preservation
            if 'compressed_accuracy' in df.columns and 'score_preservation' in df.columns:
                method_colors = {'Llmlingua2': 'red', 'Selective Context': 'blue', 'Naive Truncation': 'green'}
                if 'method' in df.columns:
                    colors = [method_colors.get(m, 'gray') for m in df['method']]
                else:
                    colors = 'blue'
                
                axes[0, 1].scatter(df['compressed_accuracy'], df['score_preservation'], 
                                 c=colors, alpha=0.7, s=50)
                axes[0, 1].set_xlabel('Compressed Accuracy (%)')
                axes[0, 1].set_ylabel('Score Preservation (%)')
                axes[0, 1].set_title('Accuracy vs Preservation')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Task Performance Ranking
            if 'task' in df.columns and 'score_preservation' in df.columns:
                task_perf = df.groupby('task')['score_preservation'].mean().sort_values(ascending=True)
                bars = axes[1, 0].barh(range(len(task_perf)), task_perf.values, 
                                     color='lightcoral', alpha=0.8)
                axes[1, 0].set_yticks(range(len(task_perf)))
                axes[1, 0].set_yticklabels(task_perf.index)
                axes[1, 0].set_xlabel('Avg Score Preservation (%)')
                axes[1, 0].set_title('Task Performance Ranking')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Efficiency Distribution
            if 'compression_efficiency' in df.columns:
                efficiency_data = df['compression_efficiency'].dropna()
                if not efficiency_data.empty:
                    axes[1, 1].hist(efficiency_data, bins=25, alpha=0.8, 
                                  color='gold', edgecolor='black')
                    axes[1, 1].set_xlabel('Compression Efficiency (%)')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].set_title('Efficiency Distribution')
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            perf_file = self.output_dir / f"performance_analysis_{timestamp}.png"
            plt.savefig(perf_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(perf_file)
            
        except Exception as e:
            logger.error(f"Error creating performance analysis chart: {e}")
            return None

    def _create_correlation_chart(self, df: pd.DataFrame, timestamp: str) -> Optional[str]:
        """Create correlation matrix chart."""
        try:
            numeric_cols = ['actual_compression_ratio', 'score_preservation', 'compressed_accuracy', 
                           'accuracy_drop', 'cost_benefit_ratio', 'compression_efficiency']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return None
            
            plt.figure(figsize=(10, 8))
            
            correlation_matrix = df[available_cols].corr()
            
            # Create heatmap with mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.3f', mask=mask, cbar_kws={"shrink": .8})
            
            plt.title('Correlation Matrix of Performance Metrics', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            corr_file = self.output_dir / f"correlation_matrix_{timestamp}.png"
            plt.savefig(corr_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(corr_file)
        
        except Exception as e:
            logger.error(f"Error creating correlation chart: {e}")
            return None

    def _create_method_task_ratio_chart(self, df: pd.DataFrame, timestamp: str) -> Optional[str]:
        """Create method performance across ratios for each task chart."""
        try:
            # Check if we have the required data
            required_cols = ['method', 'task', 'actual_preservation_ratio', 'score_preservation']
            if not all(col in df.columns for col in required_cols):
                # Fall back to old column name if new one doesn't exist yet
                if 'actual_compression_ratio' in df.columns:
                    df['actual_preservation_ratio'] = df['actual_compression_ratio']  # Temp fix for transition
                else:
                    logger.warning("Missing required columns for method-task-ratio chart")
                    return None
            
            # Create figure with subplots for each task
            tasks = df['task'].unique()
            methods = df['method'].unique()
            
            fig, axes = plt.subplots(1, len(tasks), figsize=(6*len(tasks), 8))
            if len(tasks) == 1:
                axes = [axes]
            
            fig.suptitle('Method Performance Across Preservation Ratios by Task', 
                        fontsize=16, fontweight='bold')
            
            # Define consistent colors for methods
            colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
            method_colors = dict(zip(methods, colors))
            
            for i, task in enumerate(tasks):
                ax = axes[i]
                task_data = df[df['task'] == task]
                
                # Create ratio bins for better visualization
                task_data_copy = task_data.copy()
                task_data_copy['ratio_bins'] = pd.cut(task_data_copy['actual_preservation_ratio'], 
                                                     bins=8, labels=False)
                
                # Plot each method
                for method in methods:
                    method_data = task_data_copy[task_data_copy['method'] == method]
                    if not method_data.empty:
                        # Group by ratio bins and calculate mean performance
                        ratio_perf = method_data.groupby('ratio_bins').agg({
                            'actual_preservation_ratio': 'mean',
                            'score_preservation': 'mean'
                        }).reset_index()
                        
                        if len(ratio_perf) > 1:
                            # Plot line
                            ax.plot(ratio_perf['actual_preservation_ratio'], 
                                   ratio_perf['score_preservation'],
                                   marker='o', linewidth=2, markersize=6,
                                   label=method, color=method_colors[method])
                        else:
                            # Plot single point if not enough data for line
                            ax.scatter(method_data['actual_preservation_ratio'], 
                                     method_data['score_preservation'],
                                     label=method, color=method_colors[method], s=60)
                
                ax.set_title(f'{task} Task', fontsize=14, fontweight='bold')
                ax.set_xlabel('Preservation Ratio (Target: Keep % of Original)', fontsize=12)
                ax.set_ylabel('Score Preservation (%)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Set reasonable axis limits
                if not task_data.empty:
                    ax.set_ylim(max(0, task_data['score_preservation'].min() - 5),
                               min(110, task_data['score_preservation'].max() + 5))
                    ax.set_xlim(0, 1)  # Preservation ratio goes from 0 to 1
            
            plt.tight_layout()
            
            # Save the chart
            method_task_ratio_file = self.output_dir / f"method_task_ratio_analysis_{timestamp}.png"
            plt.savefig(method_task_ratio_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Method-Task-Ratio chart saved: {method_task_ratio_file}")
            return str(method_task_ratio_file)
            
        except Exception as e:
            logger.error(f"Error creating method-task-ratio chart: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Error creating correlation chart: {e}")
            return None
