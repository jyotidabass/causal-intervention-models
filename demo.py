"""
Demo Script for Causal Intervention Models
==========================================

Quick demonstration of the causal inference system for user retention.
Run this script to see the complete system in action.
"""

import sys
import os
import time
from causal_retention import CausalRetentionAnalyzer, UserRetentionDataGenerator

def print_banner():
    """Print a nice banner for the demo."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        CAUSAL INTERVENTION MODELS FOR USER RETENTION        ║
    ║                                                              ║
    ║  🎯 Causal Inference with DoWhy & CausalML                  ║
    ║  🚀 Intervention Recommendation System                       ║
    ║  📊 A/B Testing Simulation Framework                        ║
    ║  📈 Uplift Modeling Techniques                              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_quick_demo():
    """Run a quick demonstration of the system."""
    print("🚀 Starting Quick Demo...")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n📊 Step 1: Generating synthetic user retention data...")
    generator = UserRetentionDataGenerator(n_samples=5000)
    data = generator.generate_data()
    
    # Step 2: Initialize analyzer
    print("\n🔧 Step 2: Initializing causal retention analyzer...")
    analyzer = CausalRetentionAnalyzer(data)
    
    # Step 3: Run causal analysis
    print("\n🎯 Step 3: Running causal inference analysis...")
    try:
        causal_results = analyzer.run_causal_analysis()
        if causal_results:
            print(f"   ✅ Treatment effect: {causal_results.value:.4f}")
            print(f"   ✅ P-value: {causal_results.p_value:.4f}")
        else:
            print("   ⚠️  Causal analysis skipped (DoWhy not available)")
    except Exception as e:
        print(f"   ⚠️  Causal analysis failed: {str(e)}")
    
    # Step 4: User segmentation
    print("\n👥 Step 4: Creating user segments...")
    segments = analyzer.create_user_segments()
    print(f"   ✅ Created {len(set(segments))} user segments")
    
    # Step 5: Intervention recommendations
    print("\n💡 Step 5: Generating intervention recommendations...")
    recommendations = analyzer.generate_intervention_recommendations()
    print(f"   ✅ Generated recommendations for {len(recommendations)} segments")
    
    # Step 6: A/B testing simulation
    print("\n🧪 Step 6: Running A/B testing simulation...")
    ab_results = analyzer.run_ab_test_simulation(n_tests=20, sample_size=500)
    significant_tests = ab_results['significant'].sum()
    print(f"   ✅ Simulated {len(ab_results)} A/B tests")
    print(f"   ✅ {significant_tests} tests showed significant results")
    
    # Step 7: Summary report
    print("\n📋 Step 7: Generating summary report...")
    summary = analyzer.get_summary_report()
    print("\n" + "=" * 60)
    print("📊 SUMMARY REPORT")
    print("=" * 60)
    print(summary)
    
    return analyzer

def run_uplift_demo():
    """Run uplift modeling demonstration."""
    print("\n" + "=" * 60)
    print("📈 UPLIFT MODELING DEMONSTRATION")
    print("=" * 60)
    
    # Generate data
    generator = UserRetentionDataGenerator(n_samples=8000)
    data = generator.generate_data()
    
    analyzer = CausalRetentionAnalyzer(data)
    
    # Run uplift modeling
    print("\n🔬 Running uplift modeling...")
    uplift_results = analyzer.run_uplift_modeling()
    
    if uplift_results:
        print("\n📊 Uplift Modeling Results:")
        for model_name, predictions in uplift_results.items():
            print(f"   {model_name}:")
            print(f"     Average uplift: {predictions.mean():.4f}")
            print(f"     Uplift std: {predictions.std():.4f}")
            print(f"     Max uplift: {predictions.max():.4f}")
            print(f"     Min uplift: {predictions.min():.4f}")
    else:
        print("   ⚠️  Uplift modeling skipped (CausalML not available)")
    
    return analyzer

def run_visualization_demo():
    """Run visualization demonstration."""
    print("\n" + "=" * 60)
    print("📊 VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Generate data
    generator = UserRetentionDataGenerator(n_samples=6000)
    data = generator.generate_data()
    
    analyzer = CausalRetentionAnalyzer(data)
    
    # Create segments first
    analyzer.create_user_segments()
    
    # Generate visualizations
    print("\n🎨 Creating comprehensive visualizations...")
    analyzer.create_visualizations()
    print("   ✅ Visualizations saved as 'causal_retention_analysis.png'")
    
    return analyzer

def main():
    """Main demo function."""
    print_banner()
    
    try:
        # Quick demo
        analyzer1 = run_quick_demo()
        
        # Uplift modeling demo
        analyzer2 = run_uplift_demo()
        
        # Visualization demo
        analyzer3 = run_visualization_demo()
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📁 Generated Files:")
        print("   • causal_retention_analysis.png - Comprehensive analysis plots")
        print("   • ab_test_results.png - A/B testing results (if generated)")
        
        print("\n🔧 Key Features Demonstrated:")
        print("   ✅ Synthetic data generation with realistic causal relationships")
        print("   ✅ Causal inference analysis using DoWhy")
        print("   ✅ User segmentation and targeted intervention recommendations")
        print("   ✅ A/B testing simulation framework")
        print("   ✅ Uplift modeling with multiple algorithms")
        print("   ✅ Comprehensive visualization and reporting")
        
        print("\n📚 Next Steps:")
        print("   • Modify config.py to customize parameters")
        print("   • Run example_usage.py for detailed examples")
        print("   • Integrate with your real user data")
        print("   • Deploy intervention recommendations in production")
        
        print("\n💡 Use Cases:")
        print("   • Optimize user onboarding flows")
        print("   • Design personalized retention campaigns")
        print("   • A/B test intervention strategies")
        print("   • Identify high-value user segments")
        print("   • Measure true causal effects of features")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   • Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check that you have sufficient memory for large datasets")
        print("   • Verify that matplotlib can create plots in your environment")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
