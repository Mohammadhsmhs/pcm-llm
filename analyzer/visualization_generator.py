#!/usr/bin/env python3
"""
Visualization Module for PCM-LLM Analysis
Handles creation of comprehen            # 11. Correlation Matrix
            self._plot_correlation_matrix(df, axes[2, 2])
            
            # 12. Baseline vs Compressed Performance (NEW!)
            self._plot_baseline_vs_compressed(df, axes[2, 3])
            
            plt.tight_layout()
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comprehensive dashboard saved to {viz_file}")
            return str(viz_file)zations and charts.
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
            
            # 1. Method Performance Comparison (Bar Chart)
            self._plot_method_comparison(df, axes[0, 0])
            
            # 2. Task Performance Heatmap
            self._plot_task_method_heatmap(df, axes[0, 1])
            
            # 3. Score Preservation by Compression Ratio  
            self._plot_preservation_by_ratio(df, axes[0, 2])
            
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
            
            # 12. Improvement Rate Analysis - NEW
            self._plot_improvement_rates(df, axes[2, 3])
            
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
                
            # Latency analysis dashboard (NEW!)
            latency_file = self._create_latency_analysis_dashboard(df, timestamp)
            if latency_file:
                files.append(latency_file)
            
            # Performance analysis chart
            performance_file = self._create_performance_analysis_chart(df, timestamp)
            if performance_file:
                files.append(performance_file)
            
            # Baseline analysis chart (NEW!)
            baseline_file = self._create_baseline_analysis_chart(df, timestamp)
            if baseline_file:
                files.append(baseline_file)
            
            # Correlation matrix
            correlation_file = self._create_correlation_chart(df, timestamp)
            if correlation_file:
                files.append(correlation_file)
            
            # Method-Task-Ratio analysis chart (NEW!)
            method_task_ratio_file = self._create_method_task_ratio_chart(df, timestamp)
            if method_task_ratio_file:
                files.append(method_task_ratio_file)
                
            # Detailed improvement analysis chart (NEW!)
            improvement_file = self._create_improvement_analysis_dashboard(df, timestamp)
            if improvement_file:
                files.append(improvement_file)
                
        except Exception as e:
            logger.error(f"Error creating specialized visualizations: {e}")
        
        return files

    def _plot_preservation_by_ratio(self, df: pd.DataFrame, ax):
        """Plot score preservation by compression ratio with trend line."""
        try:
            if 'actual_compression_ratio' in df.columns and 'score_preservation' in df.columns:
                # Create scatter plot
                scatter = ax.scatter(df['actual_compression_ratio'], df['score_preservation'], 
                                   c=df['cost_benefit_ratio'] if 'cost_benefit_ratio' in df.columns else 'blue',
                                   cmap='viridis', alpha=0.7, s=50)
                ax.set_xlabel('Compression Ratio')
                ax.set_ylabel('Score Preservation (%)')
                ax.set_title('Score Preservation by Compression Ratio')
                ax.grid(True, alpha=0.3)
                
                if 'cost_benefit_ratio' in df.columns:
                    plt.colorbar(scatter, ax=ax, label='Cost-Benefit Ratio')
            else:
                ax.text(0.5, 0.5, 'Missing required columns', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_method_comparison(self, df: pd.DataFrame, ax):
        """Plot method performance comparison with error bars."""
        try:
            if 'method' in df.columns and 'score_preservation' in df.columns:
                # Calculate both mean and standard deviation
                method_stats = df.groupby('method')['score_preservation'].agg(['mean', 'std', 'count']).fillna(0)
                method_stats = method_stats.sort_values('mean', ascending=False)
                
                # Calculate 95% confidence intervals
                from scipy import stats
                confidence_level = 0.95
                alpha = 1 - confidence_level
                
                # Calculate confidence intervals for each method
                conf_intervals = []
                for _, row in method_stats.iterrows():
                    if row['count'] > 1:
                        # Use t-distribution for small samples, normal for large
                        if row['count'] < 30:
                            t_val = stats.t.ppf(1 - alpha/2, row['count'] - 1)
                            margin_error = t_val * (row['std'] / np.sqrt(row['count']))
                        else:
                            z_val = stats.norm.ppf(1 - alpha/2)
                            margin_error = z_val * (row['std'] / np.sqrt(row['count']))
                        conf_intervals.append(margin_error)
                    else:
                        conf_intervals.append(0)
                
                method_stats['conf_interval'] = conf_intervals
                
                # Create bar plot with error bars
                bars = ax.bar(range(len(method_stats)), method_stats['mean'], 
                             yerr=method_stats['std'], capsize=5, alpha=0.8,
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(method_stats)],
                             error_kw={'linewidth': 1.5, 'ecolor': 'black'})
                
                # Add value labels on bars
                for i, (bar, mean_val, std_val, conf_int, count) in enumerate(zip(bars, method_stats['mean'], 
                                                                                  method_stats['std'], 
                                                                                  method_stats['conf_interval'],
                                                                                  method_stats['count'])):
                    height = bar.get_height()
                    # Main value label
                    ax.text(bar.get_x() + bar.get_width()/2, height + std_val + 2,
                           f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
                    # Sample size label
                    ax.text(bar.get_x() + bar.get_width()/2, height/2,
                           f'n={int(count)}', ha='center', va='center', fontsize=7, color='white')
                
                ax.set_xticks(range(len(method_stats)))
                ax.set_xticklabels(method_stats.index, rotation=45, ha='right')
                ax.set_title('Method Performance Comparison\n(Error bars = ±1 Std Dev)', fontsize=10)
                ax.set_ylabel('Score Preservation (%)')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add legend explaining error bars
                ax.text(0.02, 0.98, '📊 Error bars show standard deviation\n📈 Numbers show sample size', 
                       transform=ax.transAxes, fontsize=7, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
                
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
        """Plot cost-benefit vs performance scatter with outlier handling."""
        try:
            if 'cost_benefit_ratio' in df.columns and 'score_preservation' in df.columns:
                # Filter extreme outliers for better visualization (keep 95% of data)
                q5_cb = df['cost_benefit_ratio'].quantile(0.05)
                q95_cb = df['cost_benefit_ratio'].quantile(0.95)
                filtered_df = df[(df['cost_benefit_ratio'] >= q5_cb) & (df['cost_benefit_ratio'] <= q95_cb)]
                
                scatter = ax.scatter(filtered_df['cost_benefit_ratio'], filtered_df['score_preservation'], 
                                   c=filtered_df['actual_compression_ratio'] if 'actual_compression_ratio' in filtered_df.columns else 'blue',
                                   cmap='viridis', alpha=0.6, s=20)
                ax.set_xlabel('Cost-Benefit Ratio (Speed Proxy)')
                ax.set_ylabel('Score Preservation (%)')
                ax.set_title('Cost-Benefit vs Performance\n(Filtered 5-95 percentile)')
                ax.grid(True, alpha=0.3)
                
                if 'actual_compression_ratio' in filtered_df.columns:
                    plt.colorbar(scatter, ax=ax, label='Compression Ratio')
                    
                # Add summary stats as text
                total_points = len(df)
                shown_points = len(filtered_df)
                ax.text(0.02, 0.98, f'Showing {shown_points}/{total_points} points\n(5-95 percentile)', 
                       transform=ax.transAxes, va='top', ha='left', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
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
        """Plot performance trends with trend line."""
        try:
            if 'actual_compression_ratio' in df.columns and 'score_preservation' in df.columns:
                # Sample data for visualization if too many points
                sample_df = df.sample(n=min(500, len(df))) if len(df) > 500 else df
                
                # Create scatter plot
                ax.scatter(sample_df['actual_compression_ratio'], sample_df['score_preservation'], 
                          alpha=0.6, c='blue', s=20)
                
                # Add trend line using all clean data (not just sample)
                clean_df = df[['actual_compression_ratio', 'score_preservation']].dropna()
                clean_df = clean_df[~np.isinf(clean_df['actual_compression_ratio'])]
                clean_df = clean_df[~np.isinf(clean_df['score_preservation'])]
                
                if len(clean_df) >= 5:
                    try:
                        x_clean = clean_df['actual_compression_ratio']
                        y_clean = clean_df['score_preservation']
                        
                        # Try both linear and polynomial fits
                        linear_coeffs = np.polyfit(x_clean, y_clean, deg=1)
                        linear_r2 = np.corrcoef(x_clean, y_clean)[0, 1] ** 2
                        
                        # Check if polynomial provides significantly better fit
                        if len(clean_df) >= 10:
                            poly_coeffs = np.polyfit(x_clean, y_clean, deg=2)
                            poly_func = np.poly1d(poly_coeffs)
                            poly_pred = poly_func(x_clean)
                            poly_r2 = 1 - (np.sum((y_clean - poly_pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2))
                            
                            # Use polynomial if it's significantly better (R² improvement > 0.05)
                            if poly_r2 - linear_r2 > 0.05:
                                x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                                y_trend = poly_func(x_trend)
                                trend_label = f'Trend (R² = {poly_r2:.3f})'
                            else:
                                linear_func = np.poly1d(linear_coeffs)
                                x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                                y_trend = linear_func(x_trend)
                                trend_label = f'Trend (R² = {linear_r2:.3f})'
                        else:
                            # Use linear for smaller datasets
                            linear_func = np.poly1d(linear_coeffs)
                            x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                            y_trend = linear_func(x_trend)
                            trend_label = f'Trend (R² = {linear_r2:.3f})'
                        
                        # Plot trend line
                        ax.plot(x_trend, y_trend, 'red', linewidth=2, alpha=0.8, 
                               label=trend_label, linestyle='--')
                        ax.legend(fontsize=8, loc='upper right')
                        
                    except Exception as e:
                        logger.warning(f"Could not add trend line to performance trends: {e}")
                
                ax.set_xlabel('Compression Ratio')
                ax.set_ylabel('Score Preservation (%)')
                ax.set_title('Performance Trends')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Missing trend data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_ratio_achievement(self, df: pd.DataFrame, ax):
        """Plot target vs actual preservation ratios."""
        try:
            # Use actual_preservation_ratio to compare with target_ratio (both are preservation ratios)
            if 'target_ratio' in df.columns and 'actual_preservation_ratio' in df.columns:
                # Sample for visualization
                sample_df = df.sample(n=min(200, len(df))) if len(df) > 200 else df
                
                ax.scatter(sample_df['target_ratio'], sample_df['actual_preservation_ratio'], 
                          alpha=0.6, c='green', s=30)
                
                # Perfect match line
                min_val = min(sample_df['target_ratio'].min(), sample_df['actual_preservation_ratio'].min())
                max_val = max(sample_df['target_ratio'].max(), sample_df['actual_preservation_ratio'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Match')
                
                ax.set_xlabel('Target Preservation Ratio')
                ax.set_ylabel('Actual Preservation Ratio')
                ax.set_title('Target vs Actual Preservation Ratios')
                ax.legend()
            # Fall back to old column name if new one doesn't exist yet
            elif 'target_ratio' in df.columns and 'actual_compression_ratio' in df.columns:
                # Sample for visualization
                sample_df = df.sample(n=min(200, len(df))) if len(df) > 200 else df
                
                # Convert actual_compression_ratio to preservation ratio for comparison
                actual_preservation = 1 - sample_df['actual_compression_ratio']
                
                ax.scatter(sample_df['target_ratio'], actual_preservation, 
                          alpha=0.6, c='green', s=30)
                
                # Perfect match line
                min_val = min(sample_df['target_ratio'].min(), actual_preservation.min())
                max_val = max(sample_df['target_ratio'].max(), actual_preservation.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Match')
                
                ax.set_xlabel('Target Preservation Ratio')
                ax.set_ylabel('Actual Preservation Ratio (Calculated)')
                ax.set_title('Target vs Actual Preservation Ratios')
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

    def _plot_improvement_rates(self, df: pd.DataFrame, ax):
        """Plot percentage of samples that improved over baseline by method."""
        try:
            if 'score_preservation' in df.columns and 'method' in df.columns:
                methods = df['method'].unique()
                improvement_rates = []
                method_names = []
                
                for method in methods:
                    method_data = df[df['method'] == method]
                    total_samples = len(method_data)
                    improved_samples = len(method_data[method_data['score_preservation'] > 100.0])
                    improvement_rate = (improved_samples / total_samples * 100) if total_samples > 0 else 0
                    
                    improvement_rates.append(improvement_rate)
                    method_names.append(method)
                
                if improvement_rates:
                    # Create bar chart
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(methods)]
                    bars = ax.bar(method_names, improvement_rates, color=colors, alpha=0.8)
                    
                    ax.set_title('Samples Improved Over Baseline', fontweight='bold')
                    ax.set_ylabel('Improvement Rate (%)')
                    ax.set_ylim(0, max(100, max(improvement_rates) * 1.1))
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add percentage labels on bars
                    for bar, rate in zip(bars, improvement_rates):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                               f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
                    
                    # Add horizontal line at 50% for reference
                    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax.text(0.02, 52, '50% baseline', transform=ax.get_yaxis_transform(), 
                           color='red', fontsize=8, alpha=0.7)
                else:
                    ax.text(0.5, 0.5, 'No improvement data available', ha='center', va='center', 
                           transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Missing required data', ha='center', va='center', 
                       transform=ax.transAxes)
                
        except Exception as e:
            logger.warning(f"Error plotting improvement rates: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_baseline_vs_compressed(self, df: pd.DataFrame, ax):
        """Plot baseline vs compressed accuracy comparison with model awareness."""
        try:
            if 'baseline_accuracy' in df.columns and 'compressed_accuracy' in df.columns:
                # Check if we have multiple models
                if 'model' in df.columns and len(df['model'].unique()) > 1:
                    models = df['model'].unique()
                    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
                    
                    # Plot each model separately
                    for i, model in enumerate(models):
                        model_data = df[df['model'] == model]
                        short_name = model.replace('-gguf', '').replace('Phi-3-mini-4k-instruct', 'Phi-3')
                        short_name = short_name.replace('Qwen3-30B-A3B-GGUF', 'Qwen3-30B')
                        
                        scatter = ax.scatter(model_data['baseline_accuracy'], model_data['compressed_accuracy'], 
                                           c=colors[i], alpha=0.6, s=15, label=f'{short_name} (n={len(model_data)})')
                else:
                    # Single model or no model info
                    scatter = ax.scatter(df['baseline_accuracy'], df['compressed_accuracy'], 
                                       c=df['accuracy_drop'] if 'accuracy_drop' in df.columns else 'blue',
                                       cmap='RdYlBu_r', alpha=0.6, s=20)
                
                # Add diagonal line (perfect preservation)
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Preservation')
                
                # Add statistics - model-aware
                if 'model' in df.columns and len(df['model'].unique()) > 1:
                    stats_text = ""
                    for model in df['model'].unique():
                        model_data = df[df['model'] == model]
                        short_name = model.replace('-gguf', '').replace('Phi-3-mini-4k-instruct', 'Phi-3')
                        short_name = short_name.replace('Qwen3-30B-A3B-GGUF', 'Qwen3-30B')
                        
                        baseline_perfect = (model_data['baseline_accuracy'] == 1.0).sum()
                        compressed_failures = (model_data['compressed_accuracy'] == 0.0).sum()
                        perfect_to_failure = ((model_data['baseline_accuracy'] == 1.0) & (model_data['compressed_accuracy'] == 0.0)).sum()
                        
                        stats_text += f'{short_name}:\n'
                        stats_text += f'  Perfect baseline: {baseline_perfect} ({baseline_perfect/len(model_data)*100:.1f}%)\n'
                        stats_text += f'  Perfect→Fail: {perfect_to_failure}\n\n'
                else:
                    total_samples = len(df)
                    baseline_perfect = (df['baseline_accuracy'] == 1.0).sum()
                    compressed_failures = (df['compressed_accuracy'] == 0.0).sum()
                    perfect_to_failure = ((df['baseline_accuracy'] == 1.0) & (df['compressed_accuracy'] == 0.0)).sum()
                    
                    stats_text = f'Total: {total_samples} samples\n'
                    stats_text += f'Baseline Perfect: {baseline_perfect} ({baseline_perfect/total_samples*100:.1f}%)\n'
                    stats_text += f'Compressed Failures: {compressed_failures} ({compressed_failures/total_samples*100:.1f}%)\n'
                    stats_text += f'Perfect→Failure: {perfect_to_failure} ({perfect_to_failure/total_samples*100:.1f}%)'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax.set_xlabel('Baseline Accuracy')
                ax.set_ylabel('Compressed Accuracy')
                ax.set_title('Baseline vs Compressed Performance')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add colorbar only if single model and using accuracy drop colors
                if not ('model' in df.columns and len(df['model'].unique()) > 1) and 'accuracy_drop' in df.columns:
                    plt.colorbar(scatter, ax=ax, label='Accuracy Drop')
            else:
                ax.text(0.5, 0.5, 'Missing baseline/compressed data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

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
                # Remove any infinite or NaN values for trend line calculation
                clean_df = df[['actual_compression_ratio', 'score_preservation']].dropna()
                clean_df = clean_df[~np.isinf(clean_df['actual_compression_ratio'])]
                clean_df = clean_df[~np.isinf(clean_df['score_preservation'])]
                
                scatter1 = axes[0, 0].scatter(df['actual_compression_ratio'], df['score_preservation'], 
                                            c=df['cost_benefit_ratio'] if 'cost_benefit_ratio' in df.columns else 'blue',
                                            cmap='viridis', alpha=0.7, s=50)
                
                # Add trend line if we have enough clean data points
                if len(clean_df) >= 5:
                    try:
                        # Use polynomial regression (degree 2) for better fit
                        x_clean = clean_df['actual_compression_ratio']
                        y_clean = clean_df['score_preservation']
                        
                        # Try both linear and polynomial fits, use the better one
                        # Linear fit
                        linear_coeffs = np.polyfit(x_clean, y_clean, deg=1)
                        linear_r2 = np.corrcoef(x_clean, y_clean)[0,1]**2
                        
                        # Polynomial fit
                        poly_coeffs = np.polyfit(x_clean, y_clean, deg=2)
                        poly_func = np.poly1d(poly_coeffs)
                        poly_predictions = poly_func(x_clean)
                        poly_r2 = 1 - np.sum((y_clean - poly_predictions)**2) / np.sum((y_clean - y_clean.mean())**2)
                        
                        # Use the better fitting model
                        if poly_r2 > linear_r2 and poly_r2 > 0.01:
                            # Use polynomial
                            x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                            y_trend = poly_func(x_trend)
                            trend_label = f'Trend (Poly R² = {poly_r2:.3f})'
                        else:
                            # Use linear (simpler and often more interpretable)
                            linear_func = np.poly1d(linear_coeffs)
                            x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                            y_trend = linear_func(x_trend)
                            trend_label = f'Trend (Linear R² = {linear_r2:.3f})'
                        
                        # Plot trend line with enhanced visibility
                        axes[0, 0].plot(x_trend, y_trend, 'red', linewidth=3, alpha=0.9, 
                                       label=trend_label, linestyle='--')
                        axes[0, 0].legend(fontsize=10, loc='upper right')
                        
                    except Exception as e:
                        logger.warning(f"Could not add trend line: {e}")
                
                axes[0, 0].set_xlabel('Compression Ratio')
                axes[0, 0].set_ylabel('Score Preservation (%)')
                axes[0, 0].set_title('Performance vs Compression with Trend')
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
                
                # Create ratio bins for better visualization - IMPROVED binning
                task_data_copy = task_data.copy()
                # Use more bins and handle wide ranges better
                min_ratio = task_data_copy['actual_preservation_ratio'].min()
                max_ratio = task_data_copy['actual_preservation_ratio'].max()
                # Use percentile-based binning for better distribution
                bins = np.linspace(min_ratio, max_ratio, 10)
                task_data_copy['ratio_bins'] = pd.cut(task_data_copy['actual_preservation_ratio'], 
                                                     bins=bins, labels=False, duplicates='drop')
                
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
                ax.set_xlabel('Actual Preservation Ratio (1.0 = Keep 100% of Original)', fontsize=12)
                ax.set_ylabel('Score Preservation (%)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Set reasonable axis limits - FIXED to handle full data range
                if not task_data.empty:
                    # Y-axis: Allow full range of score preservation (up to 200%)
                    y_min = max(-10, task_data['score_preservation'].min() - 5)
                    y_max = min(220, task_data['score_preservation'].max() + 10)
                    ax.set_ylim(y_min, y_max)
                    
                    # X-axis: Allow full range of preservation ratios (can exceed 1.0)
                    min_ratio = task_data['actual_preservation_ratio'].min()
                    max_ratio = task_data['actual_preservation_ratio'].max()
                    x_min = max(0, min_ratio - 0.05)
                    x_max = max_ratio + 0.05  # Remove the artificial cap at 1.0
                    ax.set_xlim(x_min, x_max)
            
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

    def _create_baseline_analysis_chart(self, df: pd.DataFrame, timestamp: str) -> Optional[str]:
        """Create comprehensive baseline vs compressed performance analysis with model breakdown."""
        try:
            if 'baseline_accuracy' not in df.columns or 'compressed_accuracy' not in df.columns:
                return None
            
            # Check if we have multiple models
            models = df['model'].unique() if 'model' in df.columns else ['All Models']
            has_multiple_models = len(models) > 1 and 'model' in df.columns
            
            if has_multiple_models:
                fig, axes = plt.subplots(3, 3, figsize=(20, 16))
                fig.suptitle('Baseline Performance Analysis (Model-Specific)', fontsize=18, fontweight='bold')
            else:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('Baseline Performance Analysis', fontsize=16, fontweight='bold')
            
            # Flatten axes for easier indexing
            if has_multiple_models:
                axes_flat = axes.flatten()
            
            # 1. Overall Baseline vs Compressed Scatter (top left)
            ax_idx = 0
            if has_multiple_models:
                ax = axes_flat[ax_idx]
            else:
                ax = axes[0, 0]
                
            colors = plt.cm.Set1(np.linspace(0, 1, len(models))) if has_multiple_models else ['blue']
            
            for i, model in enumerate(models):
                if has_multiple_models:
                    model_data = df[df['model'] == model]
                    label = f'{model} (n={len(model_data)})'
                    color = colors[i]
                else:
                    model_data = df
                    label = f'All Data (n={len(model_data)})'
                    color = 'blue'
                
                scatter = ax.scatter(model_data['baseline_accuracy'], model_data['compressed_accuracy'], 
                                   c=color, alpha=0.5, s=20, label=label)
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Preservation')
            ax.set_xlabel('Baseline Accuracy')
            ax.set_ylabel('Compressed Accuracy')
            ax.set_title('Baseline vs Compressed Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Model Comparison Bar Chart (top middle)
            ax_idx += 1
            if has_multiple_models:
                ax = axes_flat[ax_idx]
                
                baseline_means = []
                compressed_means = []
                model_labels = []
                
                for model in models:
                    model_data = df[df['model'] == model]
                    baseline_means.append(model_data['baseline_accuracy'].mean())
                    compressed_means.append(model_data['compressed_accuracy'].mean())
                    # Shorten model names for display
                    short_name = model.replace('-gguf', '').replace('Phi-3-mini-4k-instruct', 'Phi-3-mini')
                    short_name = short_name.replace('Qwen3-30B-A3B-GGUF', 'Qwen3-30B')
                    model_labels.append(short_name)
                
                x = np.arange(len(models))
                width = 0.35
                
                ax.bar(x - width/2, baseline_means, width, label='Baseline', color='lightblue', alpha=0.8)
                ax.bar(x + width/2, compressed_means, width, label='Compressed', color='orange', alpha=0.8)
                
                ax.set_xlabel('Model')
                ax.set_ylabel('Mean Accuracy')
                ax.set_title('Baseline vs Compressed by Model')
                ax.set_xticks(x)
                ax.set_xticklabels(model_labels, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (baseline, compressed) in enumerate(zip(baseline_means, compressed_means)):
                    ax.text(i - width/2, baseline + 0.02, f'{baseline:.3f}', ha='center', fontsize=9)
                    ax.text(i + width/2, compressed + 0.02, f'{compressed:.3f}', ha='center', fontsize=9)
            
            # 3. Failure Analysis by Model (top right)
            ax_idx += 1
            if has_multiple_models:
                ax = axes_flat[ax_idx]
                
                failure_rates = []
                good_baseline_failure_rates = []
                
                for model in models:
                    model_data = df[df['model'] == model]
                    
                    # Overall failure rate
                    total_failures = (model_data['compressed_accuracy'] == 0.0).sum()
                    failure_rate = total_failures / len(model_data) * 100
                    failure_rates.append(failure_rate)
                    
                    # Good baseline failure rate
                    good_baseline = model_data[model_data['baseline_accuracy'] >= 0.8]
                    if len(good_baseline) > 0:
                        good_failures = (good_baseline['compressed_accuracy'] == 0.0).sum()
                        good_failure_rate = good_failures / len(good_baseline) * 100
                        good_baseline_failure_rates.append(good_failure_rate)
                    else:
                        good_baseline_failure_rates.append(0)
                
                x = np.arange(len(models))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, failure_rates, width, label='Overall Failure Rate', color='red', alpha=0.7)
                bars2 = ax.bar(x + width/2, good_baseline_failure_rates, width, label='Good Baseline Failure Rate', color='darkred', alpha=0.7)
                
                ax.set_xlabel('Model')
                ax.set_ylabel('Failure Rate (%)')
                ax.set_title('Compression Failure Rates by Model')
                ax.set_xticks(x)
                ax.set_xticklabels(model_labels, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, rate in zip(bars1, failure_rates):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.1f}%', ha='center', fontsize=9)
                for bar, rate in zip(bars2, good_baseline_failure_rates):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.1f}%', ha='center', fontsize=9)
            
            # 4-6. Model-specific scatter plots (if multiple models)
            if has_multiple_models and len(models) >= 2:
                for i, model in enumerate(models[:2]):  # Show first 2 models
                    ax_idx += 1
                    ax = axes_flat[ax_idx + i]
                    
                    model_data = df[df['model'] == model]
                    
                    scatter = ax.scatter(model_data['baseline_accuracy'], model_data['compressed_accuracy'], 
                                       c=model_data['accuracy_drop'] if 'accuracy_drop' in df.columns else 'blue',
                                       cmap='RdYlBu_r', alpha=0.6, s=20)
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=1)
                    
                    # Add stats
                    perfect_baseline = model_data[model_data['baseline_accuracy'] == 1.0]
                    if len(perfect_baseline) > 0:
                        perfect_failures = (perfect_baseline['compressed_accuracy'] == 0.0).sum()
                        stats_text = f'Perfect Baseline: {len(perfect_baseline)}\n'
                        stats_text += f'Failed: {perfect_failures} ({perfect_failures/len(perfect_baseline)*100:.1f}%)'
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    short_name = model.replace('-gguf', '').replace('Phi-3-mini-4k-instruct', 'Phi-3-mini')
                    short_name = short_name.replace('Qwen3-30B-A3B-GGUF', 'Qwen3-30B')
                    ax.set_title(f'{short_name}')
                    ax.set_xlabel('Baseline Accuracy')
                    ax.set_ylabel('Compressed Accuracy')
                    ax.grid(True, alpha=0.3)
                    
                    if 'accuracy_drop' in df.columns:
                        plt.colorbar(scatter, ax=ax, label='Accuracy Drop')
                
                # 7. Method effectiveness by model (bottom left)
                ax_idx = 6
                ax = axes_flat[ax_idx]
                
                if 'method' in df.columns:
                    methods = df['method'].unique()
                    model_names = [m.replace('-gguf', '').replace('Phi-3-mini-4k-instruct', 'Phi-3-mini').replace('Qwen3-30B-A3B-GGUF', 'Qwen3-30B') for m in models]
                    
                    # Create effectiveness matrix
                    effectiveness_matrix = []
                    for model in models:
                        model_effectiveness = []
                        for method in methods:
                            model_method_data = df[(df['model'] == model) & (df['method'] == method)]
                            perfect_baseline = model_method_data[model_method_data['baseline_accuracy'] == 1.0]
                            
                            if len(perfect_baseline) > 0:
                                maintained = (perfect_baseline['compressed_accuracy'] == 1.0).sum()
                                effectiveness = maintained / len(perfect_baseline) * 100
                                model_effectiveness.append(effectiveness)
                            else:
                                model_effectiveness.append(0)
                        effectiveness_matrix.append(model_effectiveness)
                    
                    # Create heatmap
                    im = ax.imshow(effectiveness_matrix, cmap='RdYlGn', aspect='auto')
                    
                    # Add text annotations
                    for i in range(len(models)):
                        for j in range(len(methods)):
                            text = ax.text(j, i, f'{effectiveness_matrix[i][j]:.1f}%',
                                         ha="center", va="center", color="black", fontweight='bold')
                    
                    ax.set_xticks(range(len(methods)))
                    ax.set_yticks(range(len(models)))
                    ax.set_xticklabels([m.replace(' ', '\n') for m in methods], rotation=45)
                    ax.set_yticklabels(model_names)
                    ax.set_title('Perfect Baseline Preservation Rate by Model & Method')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Success Rate (%)')
                
                # 8. Accuracy drop distribution by model (bottom middle)
                ax_idx = 7
                ax = axes_flat[ax_idx]
                
                if 'accuracy_drop' in df.columns:
                    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
                    for i, model in enumerate(models):
                        model_data = df[df['model'] == model]
                        drop_data = model_data['accuracy_drop'].dropna()
                        short_name = model.replace('-gguf', '').replace('Phi-3-mini-4k-instruct', 'Phi-3-mini')
                        short_name = short_name.replace('Qwen3-30B-A3B-GGUF', 'Qwen3-30B')
                        ax.hist(drop_data, bins=20, alpha=0.6, label=f'{short_name} (μ={drop_data.mean():.3f})', color=colors[i])
                    
                    ax.axvline(0, color='green', linestyle='-', alpha=0.7, label='No Change')
                    ax.set_xlabel('Accuracy Drop')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Accuracy Drop Distribution by Model')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No accuracy drop data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Accuracy Drop Distribution')
                
                # 9. Summary statistics (bottom right)
                ax_idx = 8
                ax = axes_flat[ax_idx]
                
                summary_text = "MODEL COMPARISON SUMMARY\n\n"
                for model in models:
                    model_data = df[df['model'] == model]
                    short_name = model.replace('-gguf', '').replace('Phi-3-mini-4k-instruct', 'Phi-3-mini')
                    short_name = short_name.replace('Qwen3-30B-A3B-GGUF', 'Qwen3-30B')
                    
                    summary_text += f"{short_name}:\n"
                    summary_text += f"• Samples: {len(model_data)}\n"
                    summary_text += f"• Baseline μ: {model_data['baseline_accuracy'].mean():.3f}\n"
                    summary_text += f"• Compressed μ: {model_data['compressed_accuracy'].mean():.3f}\n"
                    summary_text += f"• Drop μ: {model_data['accuracy_drop'].mean():.3f}\n"
                    
                    perfect_baseline = model_data[model_data['baseline_accuracy'] == 1.0]
                    if len(perfect_baseline) > 0:
                        failures = (perfect_baseline['compressed_accuracy'] == 0.0).sum()
                        summary_text += f"• Perfect→Fail: {failures}/{len(perfect_baseline)} ({failures/len(perfect_baseline)*100:.1f}%)\n"
                    
                    summary_text += "\n"
                
                ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('Summary Statistics')
                
            else:
                # Original 2x2 layout for single model case
                # ... (keep existing non-model-specific code for 2x2 layout)
                pass
            
            plt.tight_layout()
            
            # Save the chart
            baseline_file = self.output_dir / f"baseline_analysis_{timestamp}.png"
            plt.savefig(baseline_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model-aware baseline analysis chart saved: {baseline_file}")
            return str(baseline_file)
            
        except Exception as e:
            logger.error(f"Error creating baseline analysis chart: {e}")
            return None

    def _create_latency_analysis_dashboard(self, df: pd.DataFrame, timestamp: str) -> str:
        """Create comprehensive latency analysis dashboard."""
        try:
            # Set up the dashboard (2x2 grid)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('📈 Latency & Performance Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Per-Method Latency Patterns (Top-Left)
            self._plot_method_latency_patterns(df, axes[0, 0])
            
            # 2. Task-Specific Speed Differences (Top-Right)  
            self._plot_task_speed_differences(df, axes[0, 1])
            
            # 3. Compression Ratio Impact on Processing Time (Bottom-Left)
            self._plot_compression_ratio_impact(df, axes[1, 0])
            
            # 4. Quality vs Speed Trade-offs (Bottom-Right)
            self._plot_quality_speed_tradeoffs(df, axes[1, 1])
            
            plt.tight_layout()
            
            # Save the dashboard
            latency_file = self.output_dir / f"latency_analysis_dashboard_{timestamp}.png"
            plt.savefig(latency_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Latency analysis dashboard saved: {latency_file}")
            return str(latency_file)
            
        except Exception as e:
            logger.error(f"Error creating latency analysis dashboard: {e}")
            return None

    def _plot_method_latency_patterns(self, df: pd.DataFrame, ax):
        """Plot per-method latency patterns using actual latency data or efficiency."""
        try:
            # Try to use actual latency data first
            latency_cols = ['compressed_latency', 'latency_overhead', 'latency_overhead_percent']
            latency_col = None
            
            for col in latency_cols:
                if col in df.columns and not df[col].isna().all():
                    latency_col = col
                    break
            
            if latency_col and 'method' in df.columns:
                # Use actual latency data
                filtered_df = df[df[latency_col].notna()]
                
                # Remove extreme outliers
                q1, q99 = filtered_df[latency_col].quantile([0.01, 0.99])
                filtered_df = filtered_df[(filtered_df[latency_col] >= q1) & (filtered_df[latency_col] <= q99)]
                
                method_data = []
                method_names = []
                
                for method in filtered_df['method'].unique():
                    method_subset = filtered_df[filtered_df['method'] == method]
                    method_data.append(method_subset[latency_col])
                    method_names.append(method)
                
                bp = ax.boxplot(method_data, labels=method_names, patch_artist=True)
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for patch, color in zip(bp['boxes'], colors[:len(method_names)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ylabel = 'Latency (ms)' if 'latency' in latency_col.lower() else 'Overhead (%)'
                ax.set_title(f'Per-Method Latency Patterns\n({latency_col.replace("_", " ").title()})', fontweight='bold')
                ax.set_ylabel(ylabel)
                
            elif 'method' in df.columns and 'cost_benefit_ratio' in df.columns:
                # Fallback to cost-benefit as efficiency metric (not latency)
                q99 = df['cost_benefit_ratio'].quantile(0.99)
                filtered_df = df[df['cost_benefit_ratio'] <= q99]
                
                method_data = []
                method_names = []
                
                for method in filtered_df['method'].unique():
                    method_subset = filtered_df[filtered_df['method'] == method]
                    method_data.append(method_subset['cost_benefit_ratio'])
                    method_names.append(method)
                
                bp = ax.boxplot(method_data, labels=method_names, patch_artist=True)
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for patch, color in zip(bp['boxes'], colors[:len(method_names)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title('Per-Method Efficiency Patterns\n(Higher = Better Accuracy/Compression)', fontweight='bold')
                ax.set_ylabel('Cost-Benefit Ratio (Efficiency)')
            else:
                ax.text(0.5, 0.5, 'No latency data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Latency Analysis - No Data', fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
                
        except Exception as e:
            logger.warning(f"Error plotting method latency patterns: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_task_speed_differences(self, df: pd.DataFrame, ax):
        """Plot task-specific speed differences."""
        try:
            if 'task' in df.columns and 'method' in df.columns and 'cost_benefit_ratio' in df.columns:
                # Create heatmap of speed by task and method
                pivot = df.groupby(['method', 'task'])['cost_benefit_ratio'].mean().unstack()
                
                if not pivot.empty:
                    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
                    
                    # Set ticks and labels
                    ax.set_xticks(range(len(pivot.columns)))
                    ax.set_yticks(range(len(pivot.index)))
                    ax.set_xticklabels(pivot.columns)
                    ax.set_yticklabels(pivot.index)
                    
                    # Add text annotations
                    for i in range(len(pivot.index)):
                        for j in range(len(pivot.columns)):
                            value = pivot.iloc[i, j]
                            if not np.isnan(value):
                                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                                       color='white' if value < pivot.values.mean() else 'black')
                    
                    ax.set_title('Task-Specific Speed Differences\n(Higher = Faster)', fontweight='bold')
                    ax.set_xlabel('Task')
                    ax.set_ylabel('Method')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Speed (Cost-Benefit Ratio)')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Missing task/method data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_compression_ratio_impact(self, df: pd.DataFrame, ax):
        """Plot compression ratio impact on processing time."""
        try:
            if 'actual_compression_ratio' in df.columns and 'cost_benefit_ratio' in df.columns:
                # Create bins for compression ratios
                df_copy = df.copy()
                df_copy['ratio_bins'] = pd.cut(df_copy['actual_compression_ratio'], bins=5, precision=2)
                
                # Calculate mean speed for each ratio bin
                ratio_speed = df_copy.groupby('ratio_bins')['cost_benefit_ratio'].agg(['mean', 'std']).reset_index()
                ratio_speed['ratio_midpoint'] = ratio_speed['ratio_bins'].apply(lambda x: x.mid)
                
                # Create bar plot with error bars
                bars = ax.bar(range(len(ratio_speed)), ratio_speed['mean'], 
                             yerr=ratio_speed['std'], capsize=5, alpha=0.7, color='skyblue')
                
                ax.set_xticks(range(len(ratio_speed)))
                ax.set_xticklabels([f'{interval.left:.2f}-{interval.right:.2f}' for interval in ratio_speed['ratio_bins']], 
                                  rotation=45)
                ax.set_title('Compression Ratio Impact on Processing Speed', fontweight='bold')
                ax.set_xlabel('Compression Ratio Range')
                ax.set_ylabel('Speed (Cost-Benefit Ratio)')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, mean_val in zip(bars, ratio_speed['mean']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{mean_val:.2f}', ha='center', va='bottom')
                
            else:
                ax.text(0.5, 0.5, 'Missing compression/speed data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_quality_speed_tradeoffs(self, df: pd.DataFrame, ax):
        """Plot quality vs speed trade-offs with better scale handling."""
        try:
            if 'cost_benefit_ratio' in df.columns and 'score_preservation' in df.columns:
                # Filter extreme outliers for better visualization
                q1_cb = df['cost_benefit_ratio'].quantile(0.01)
                q99_cb = df['cost_benefit_ratio'].quantile(0.99)
                filtered_df = df[(df['cost_benefit_ratio'] >= q1_cb) & (df['cost_benefit_ratio'] <= q99_cb)]
                
                # Create scatter plot colored by method
                methods = filtered_df['method'].unique()
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                for i, method in enumerate(methods):
                    method_data = filtered_df[filtered_df['method'] == method]
                    ax.scatter(method_data['cost_benefit_ratio'], method_data['score_preservation'],
                              alpha=0.6, s=30, c=colors[i % len(colors)], label=method)
                
                ax.set_xlabel('Speed (Cost-Benefit Ratio)')
                ax.set_ylabel('Quality (Score Preservation %)')
                ax.set_title('Quality vs Speed Trade-offs\n(Upper-Right = Best)', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add quadrant labels
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ax.text(xlim[1] * 0.8, ylim[1] * 0.9, 'High Quality\nHigh Speed', 
                       ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                ax.text(xlim[0] * 1.2, ylim[0] * 1.1, 'Low Quality\nLow Speed', 
                       ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                
            else:
                ax.text(0.5, 0.5, 'Missing quality/speed data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _create_improvement_analysis_dashboard(self, df: pd.DataFrame, timestamp: str) -> Optional[str]:
        """Create detailed improvement analysis dashboard with multi-dimensional breakdown."""
        try:
            improvement_file = self.output_dir / f"improvement_analysis_dashboard_{timestamp}.png"
            
            # Create comprehensive improvement analysis
            fig, axes = plt.subplots(2, 3, figsize=(21, 12))
            fig.suptitle('Detailed Improvement Analysis - When Compression Beats Baseline', fontsize=18, fontweight='bold')
            
            # Filter to only cases where compressed > baseline
            improved_cases = df[df['score_preservation'] > 100.0].copy()
            
            if len(improved_cases) > 0:
                # 1. Improvement by Compression Ratio
                self._plot_improvement_by_ratio(improved_cases, df, axes[0, 0])
                
                # 2. Improvement by Method and Model
                self._plot_improvement_by_method_model(improved_cases, df, axes[0, 1])
                
                # 3. Cost-Benefit of Improvement Cases
                self._plot_improvement_cost_benefit(improved_cases, axes[0, 2])
                
                # 4. Improvement Distribution by Task
                self._plot_improvement_by_task(improved_cases, df, axes[1, 0])
                
                # 5. Improvement Magnitude Analysis
                self._plot_improvement_magnitude(improved_cases, axes[1, 1])
                
                # 6. Improvement Summary Statistics
                self._plot_improvement_summary(improved_cases, df, axes[1, 2])
                
            else:
                # No improvement cases found
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'No improvement cases found', ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            plt.savefig(improvement_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Improvement analysis dashboard saved: {improvement_file}")
            return str(improvement_file)
            
        except Exception as e:
            logger.error(f"Error creating improvement analysis dashboard: {e}")
            return None

    def _plot_improvement_by_ratio(self, improved_cases: pd.DataFrame, all_data: pd.DataFrame, ax):
        """Plot improvement rate by compression ratio bins."""
        try:
            if 'actual_compression_ratio' in all_data.columns:
                # Create ratio bins
                ratio_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
                all_data['ratio_bin'] = pd.cut(all_data['actual_compression_ratio'], bins=ratio_bins)
                improved_cases['ratio_bin'] = pd.cut(improved_cases['actual_compression_ratio'], bins=ratio_bins)
                
                # Calculate improvement rates per bin
                bin_rates = []
                bin_labels = []
                
                for bin_val in all_data['ratio_bin'].cat.categories:
                    total_in_bin = len(all_data[all_data['ratio_bin'] == bin_val])
                    improved_in_bin = len(improved_cases[improved_cases['ratio_bin'] == bin_val])
                    rate = (improved_in_bin / total_in_bin * 100) if total_in_bin > 0 else 0
                    
                    bin_rates.append(rate)
                    bin_labels.append(f'{bin_val.left:.1f}-{bin_val.right:.1f}')
                
                bars = ax.bar(bin_labels, bin_rates, color='skyblue', alpha=0.8)
                ax.set_title('Improvement Rate by Compression Ratio', fontweight='bold')
                ax.set_xlabel('Compression Ratio (fraction removed)')
                ax.set_ylabel('Improvement Rate (%)')
                ax.tick_params(axis='x', rotation=45)
                
                # Add percentage labels on bars
                for bar, rate in zip(bars, bin_rates):
                    if rate > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                               f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No compression ratio data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_improvement_by_method_model(self, improved_cases: pd.DataFrame, all_data: pd.DataFrame, ax):
        """Plot improvement breakdown by method and model."""
        try:
            if 'method' in all_data.columns and 'model' in all_data.columns:
                # Calculate improvement rates by method-model combination
                method_model_stats = []
                
                for method in all_data['method'].unique():
                    for model in all_data['model'].unique():
                        subset_all = all_data[(all_data['method'] == method) & (all_data['model'] == model)]
                        subset_improved = improved_cases[(improved_cases['method'] == method) & (improved_cases['model'] == model)]
                        
                        total = len(subset_all)
                        improved = len(subset_improved)
                        rate = (improved / total * 100) if total > 0 else 0
                        
                        # Shorten model names for display
                        short_model = model.replace('-gguf', '').replace('Phi-3-mini-4k-instruct', 'Phi-3')
                        short_model = short_model.replace('Qwen3-30B-A3B-GGUF', 'Qwen3')
                        
                        method_model_stats.append({
                            'method': method,
                            'model': short_model,
                            'rate': rate,
                            'improved': improved,
                            'total': total
                        })
                
                # Create grouped bar chart
                df_stats = pd.DataFrame(method_model_stats)
                pivot = df_stats.pivot(index='method', columns='model', values='rate').fillna(0)
                
                pivot.plot(kind='bar', ax=ax, color=['lightcoral', 'lightblue'], alpha=0.8)
                ax.set_title('Improvement Rate by Method & Model', fontweight='bold')
                ax.set_ylabel('Improvement Rate (%)')
                ax.set_xlabel('Compression Method')
                ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.tick_params(axis='x', rotation=45)
                
            else:
                ax.text(0.5, 0.5, 'Missing method/model data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_improvement_cost_benefit(self, improved_cases: pd.DataFrame, ax):
        """Plot cost-benefit distribution of improvement cases."""
        try:
            if 'cost_benefit_ratio' in improved_cases.columns and len(improved_cases) > 0:
                # Remove extreme outliers for better visualization
                q1, q99 = improved_cases['cost_benefit_ratio'].quantile([0.01, 0.99])
                filtered = improved_cases[(improved_cases['cost_benefit_ratio'] >= q1) & 
                                        (improved_cases['cost_benefit_ratio'] <= q99)]
                
                if len(filtered) > 0:
                    ax.hist(filtered['cost_benefit_ratio'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
                    ax.set_title('Cost-Benefit Distribution\n(Improvement Cases Only)', fontweight='bold')
                    ax.set_xlabel('Cost-Benefit Ratio (Efficiency)')
                    ax.set_ylabel('Number of Cases')
                    
                    # Add statistics
                    mean_cb = filtered['cost_benefit_ratio'].mean()
                    ax.axvline(mean_cb, color='red', linestyle='--', alpha=0.8)
                    ax.text(0.7, 0.9, f'Mean: {mean_cb:.2f}', transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No valid cost-benefit data', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No cost-benefit data available', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_improvement_by_task(self, improved_cases: pd.DataFrame, all_data: pd.DataFrame, ax):
        """Plot improvement rate by task type."""
        try:
            if 'task' in all_data.columns:
                task_rates = []
                task_names = []
                
                for task in all_data['task'].unique():
                    total_task = len(all_data[all_data['task'] == task])
                    improved_task = len(improved_cases[improved_cases['task'] == task])
                    rate = (improved_task / total_task * 100) if total_task > 0 else 0
                    
                    task_rates.append(rate)
                    task_names.append(task.capitalize())
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(task_names)]
                bars = ax.bar(task_names, task_rates, color=colors, alpha=0.8)
                
                ax.set_title('Improvement Rate by Task Type', fontweight='bold')
                ax.set_ylabel('Improvement Rate (%)')
                ax.set_xlabel('Task Type')
                
                # Add percentage labels
                for bar, rate in zip(bars, task_rates):
                    if rate > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                               f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
                        
            else:
                ax.text(0.5, 0.5, 'No task data available', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_improvement_magnitude(self, improved_cases: pd.DataFrame, ax):
        """Plot distribution of improvement magnitudes."""
        try:
            if 'score_preservation' in improved_cases.columns and len(improved_cases) > 0:
                # Calculate improvement magnitude (how much better than 100%)
                improvements = improved_cases['score_preservation'] - 100.0
                
                ax.hist(improvements, bins=15, color='gold', alpha=0.7, edgecolor='black')
                ax.set_title('Improvement Magnitude Distribution', fontweight='bold')
                ax.set_xlabel('Improvement Over Baseline (%)')
                ax.set_ylabel('Number of Cases')
                
                # Add statistics
                mean_imp = improvements.mean()
                max_imp = improvements.max()
                ax.axvline(mean_imp, color='red', linestyle='--', alpha=0.8)
                ax.text(0.6, 0.9, f'Mean: +{mean_imp:.1f}%\nMax: +{max_imp:.1f}%', 
                       transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No improvement data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def _plot_improvement_summary(self, improved_cases: pd.DataFrame, all_data: pd.DataFrame, ax):
        """Plot key summary statistics for improvements."""
        try:
            # Calculate key metrics
            total_cases = len(all_data)
            improved_count = len(improved_cases)
            improvement_rate = (improved_count / total_cases * 100) if total_cases > 0 else 0
            
            if len(improved_cases) > 0:
                avg_improvement = (improved_cases['score_preservation'] - 100.0).mean()
                max_improvement = (improved_cases['score_preservation'] - 100.0).max()
                
                # Best compression ratio for improvements
                if 'actual_compression_ratio' in improved_cases.columns:
                    best_ratio = improved_cases.loc[improved_cases['score_preservation'].idxmax(), 'actual_compression_ratio']
                    best_ratio_pct = best_ratio * 100
                else:
                    best_ratio_pct = 0
                
                # Create text summary
                summary_text = f"""IMPROVEMENT SUMMARY
                
Total Cases: {total_cases:,}
Improved Cases: {improved_count:,}
Improvement Rate: {improvement_rate:.1f}%

Average Improvement: +{avg_improvement:.1f}%
Maximum Improvement: +{max_improvement:.1f}%
Best Case Compression: {best_ratio_pct:.1f}%

Key Finding: Compression improves 
performance in {improvement_rate:.1f}% of cases"""

                ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No improvement cases found', ha='center', va='center', transform=ax.transAxes)
                
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Summary Statistics', fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
