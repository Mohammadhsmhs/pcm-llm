#!/usr/bin/env python3
"""
Refactored Ultra-Comprehensive Multi-Ratio Benchmark Analyzer for PCM-LLM
Clean, modular architecture with separated concerns for maintainability.
"""

import sys
import argparse
from pathlib import Path
import logging
from typing import Optional

# Add the analyzer package to path
sys.path.insert(0, str(Path(__file__).parent))

from analyzer import UltraComprehensiveBenchmarkAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analyzer.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run the ultra-comprehensive analysis."""
    
    parser = argparse.ArgumentParser(
        description="Ultra-Comprehensive PCM-LLM Benchmark Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run full analysis on 'results' directory
  %(prog)s --results-dir /path/to/results  # Specify custom results directory
  %(prog)s --task classification    # Analyze only classification tasks
  %(prog)s --method llmlingua2,selective_context  # Compare specific methods
  %(prog)s --summary               # Show data summary without full analysis
  %(prog)s --no-visualizations     # Skip visualization generation
        """
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='results',
        help='Directory containing benchmark results (default: results)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'reasoning', 'summarization'],
        help='Analyze only a specific task'
    )
    
    parser.add_argument(
        '--methods',
        type=str,
        help='Comma-separated list of methods to compare (e.g., llmlingua2,selective_context)'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show data summary only, skip full analysis'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true', 
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_output',
        help='Directory for output files (default: analysis_output)'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        print(f"🔧 Initializing PCM-LLM Ultra-Comprehensive Analyzer v2.0")
        print(f"📂 Results directory: {args.results_dir}")
        print(f"📁 Output directory: {args.output_dir}")
        
        analyzer = UltraComprehensiveBenchmarkAnalyzer(results_dir=args.results_dir)
        
        # Set output directory for report generator and visualizer
        analyzer.report_generator.output_dir = Path(args.output_dir)
        analyzer.visualization_generator.output_dir = Path(args.output_dir)
        
        # Create output directory
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # Handle different run modes
        if args.summary:
            print("\n📊 Generating data summary...")
            summary = analyzer.get_data_summary()
            
            if summary:
                print(f"\n📈 Data Summary:")
                print(f"   • Total files: {summary.get('total_files', 0)}")
                print(f"   • Total samples: {summary.get('total_samples', 0):,}")
                print(f"   • Methods found: {', '.join(summary.get('methods', set()))}")
                print(f"   • Models found: {', '.join(summary.get('models', set()))}")
                
                for task, stats in summary.get('tasks', {}).items():
                    print(f"   • {task.title()}: {stats['files']} files, {stats['samples']} samples")
            else:
                print("❌ No data found or error generating summary")
                return 1
                
        elif args.task:
            print(f"\n🎯 Analyzing specific task: {args.task}")
            result = analyzer.analyze_specific_task(args.task)
            
            if result:
                print(f"✅ Task analysis complete!")
                print(f"   • Samples: {result['summary']['samples']}")
                print(f"   • Methods: {result['summary']['methods']}")
                print(f"   • Ratios: {result['summary']['ratios']}")
            else:
                print(f"❌ No data found for task: {args.task}")
                return 1
                
        elif args.methods:
            methods = [m.strip() for m in args.methods.split(',')]
            print(f"\n⚔️  Comparing methods: {', '.join(methods)}")
            result = analyzer.compare_methods(methods)
            
            if result:
                print(f"✅ Method comparison complete!")
                print(f"   • Total samples: {result['summary']['total_samples']}")
                for method, count in result['summary']['samples_per_method'].items():
                    print(f"   • {method}: {count} samples")
            else:
                print(f"❌ No data found for methods: {methods}")
                return 1
                
        else:
            # Full comprehensive analysis
            print(f"\n🚀 Running full ultra-comprehensive analysis...")
            
            if args.no_visualizations:
                analyzer.visualization_generator.available = False
                print("📊 Visualizations disabled")
            
            results = analyzer.run_ultra_comprehensive_analysis()
            
            if results:
                summary = results['summary']
                print(f"\n🎉 Analysis completed successfully!")
                print(f"   📊 {summary['total_samples']:,} samples analyzed")
                print(f"   📈 {summary['unique_ratios']:,} unique compression ratios")
                print(f"   🔧 {summary['methods']} methods tested")
                print(f"   📝 {summary['tasks']} tasks evaluated")
                print(f"   🤖 {summary['models']} models used")
                print(f"\n📄 Output files:")
                print(f"   • Report: {Path(results['report_file']).name}")
                print(f"   • Data CSV: {Path(results['csv_file']).name}")
                
                if results.get('visualization_files'):
                    print(f"   • Visualizations: {len(results['visualization_files'])} files")
                    for viz_file in results['visualization_files']:
                        print(f"     - {Path(viz_file).name}")
            else:
                print("❌ Analysis failed - check logs for details")
                return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    print(f"\n✅ PCM-LLM Analysis completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
