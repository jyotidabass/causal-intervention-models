"""
Example Usage of Causal Intervention Models for User Retention
============================================================

This script demonstrates how to use the causal retention analyzer
for various scenarios and use cases.
"""

import pandas as pd
import numpy as np
from causal_retention import CausalRetentionAnalyzer, UserRetentionDataGenerator
import matplotlib.pyplot as plt

def example_1_basic_analysis():
    """Example 1: Basic causal inference analysis."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Causal Inference Analysis")
    print("=" * 60)
    
    # Generate synthetic data
    generator = UserRetentionDataGenerator(n_samples=5000)
    data = generator.generate_data()
    
    # Initialize analyzer
    analyzer = CausalRetentionAnalyzer()
    analyzer.load_data(data)
    
    # Run causal analysis
    causal_results = analyzer.run_causal_analysis()
    
    # Print results
    print(f"Causal effect estimate: {causal_results.value:.4f}")
    print(f"Confidence interval: {causal_results.get_confidence_intervals()}")
    print(f"P-value: {causal_results.p_value:.4f}")
    
    return analyzer

def example_2_uplift_modeling():
    """Example 2: Advanced uplift modeling."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Uplift Modeling")
    print("=" * 60)
    
    # Use data from previous example
    generator = UserRetentionDataGenerator(n_samples=8000)
    data = generator.generate_data()
    
    analyzer = CausalRetentionAnalyzer(data)
    
    # Run uplift modeling
    uplift_results = analyzer.run_uplift_modeling()
    
    if uplift_results:
        print("Uplift modeling results:")
        for model_name, predictions in uplift_results.items():
            print(f"{model_name}:")
            print(f"  Average uplift: {predictions.mean():.4f}")
            print(f"  Uplift std: {predictions.std():.4f}")
    
    return analyzer

def example_3_user_segmentation():
    """Example 3: User segmentation and targeted interventions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: User Segmentation & Targeted Interventions")
    print("=" * 60)
    
    generator = UserRetentionDataGenerator(n_samples=10000)
    data = generator.generate_data()
    
    analyzer = CausalRetentionAnalyzer(data)
    
    # Create user segments
    segments = analyzer.create_user_segments()
    
    # Generate intervention recommendations
    recommendations = analyzer.generate_intervention_recommendations()
    
    # Display recommendations
    for segment, rec in recommendations.items():
        print(f"\nSegment {segment} ({rec['priority']} priority):")
        print(f"  Retention rate: {rec['retention_rate']:.3f}")
        print(f"  Expected impact: {rec['expected_impact']}")
        print("  Top interventions:")
        for intervention in rec['interventions'][:3]:
            print(f"    - {intervention}")
    
    return analyzer

def example_4_ab_testing():
    """Example 4: A/B testing simulation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: A/B Testing Simulation")
    print("=" * 60)
    
    generator = UserRetentionDataGenerator(n_samples=12000)
    data = generator.generate_data()
    
    analyzer = CausalRetentionAnalyzer(data)
    
    # Run A/B test simulation
    ab_results = analyzer.run_ab_test_simulation(n_tests=50, sample_size=2000)
    
    # Analyze results
    print(f"A/B Test Simulation Summary:")
    print(f"  Average lift: {ab_results['lift'].mean():.4f}")
    print(f"  Significant tests: {ab_results['significant'].sum()}/{len(ab_results)}")
    print(f"  Statistical power: {ab_results['significant'].mean():.3f}")
    print(f"  Average p-value: {ab_results['p_value'].mean():.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(ab_results['lift'], bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of A/B Test Lifts')
    plt.xlabel('Lift (Treatment - Control)')
    plt.ylabel('Frequency')
    plt.axvline(ab_results['lift'].mean(), color='red', linestyle='--', 
                label=f'Mean: {ab_results["lift"].mean():.4f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    significant_lifts = ab_results[ab_results['significant']]['lift']
    non_significant_lifts = ab_results[~ab_results['significant']]['lift']
    
    plt.hist([significant_lifts, non_significant_lifts], bins=15, alpha=0.7, 
             label=['Significant', 'Non-significant'], color=['green', 'red'])
    plt.title('Lift Distribution by Significance')
    plt.xlabel('Lift (Treatment - Control)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ab_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return analyzer

def example_5_comprehensive_analysis():
    """Example 5: Complete end-to-end analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Comprehensive End-to-End Analysis")
    print("=" * 60)
    
    # Generate larger dataset for comprehensive analysis
    generator = UserRetentionDataGenerator(n_samples=15000)
    data = generator.generate_data()
    
    analyzer = CausalRetentionAnalyzer(data)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Generate and display summary report
    summary = analyzer.get_summary_report()
    print(summary)
    
    return analyzer, results

def example_6_custom_scenarios():
    """Example 6: Custom scenarios and advanced usage."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Custom Scenarios")
    print("=" * 60)
    
    # Scenario 1: High-value users analysis
    print("Scenario 1: High-Value Users Analysis")
    generator = UserRetentionDataGenerator(n_samples=8000)
    data = generator.generate_data()
    
    # Define high-value users (top 20% by income and time spent)
    data['high_value'] = (
        (data['income'] >= data['income'].quantile(0.8)) & 
        (data['time_spent'] >= data['time_spent'].quantile(0.8))
    ).astype(int)
    
    print(f"High-value users: {data['high_value'].sum()}/{len(data)} ({data['high_value'].mean():.1%})")
    print(f"High-value retention: {data[data['high_value']==1]['retention'].mean():.3f}")
    print(f"Regular users retention: {data[data['high_value']==0]['retention'].mean():.3f}")
    
    # Scenario 2: Different intervention types
    print("\nScenario 2: Multiple Intervention Types")
    
    # Create different intervention types
    np.random.seed(42)
    data['intervention_type'] = np.random.choice(
        ['none', 'premium', 'support', 'features', 'gamification'], 
        len(data), 
        p=[0.4, 0.2, 0.15, 0.15, 0.1]
    )
    
    # Calculate retention by intervention type
    intervention_retention = data.groupby('intervention_type')['retention'].agg(['mean', 'count'])
    print("\nRetention by Intervention Type:")
    print(intervention_retention)
    
    # Scenario 3: Time-based analysis
    print("\nScenario 3: Time-based Retention Analysis")
    
    # Simulate user cohort data
    data['signup_date'] = pd.date_range('2023-01-01', periods=len(data), freq='D')
    data['cohort'] = data['signup_date'].dt.to_period('M')
    
    # Calculate cohort retention
    cohort_retention = data.groupby('cohort')['retention'].agg(['mean', 'count'])
    print("\nCohort Retention Rates:")
    print(cohort_retention.head(10))
    
    return data

def main():
    """Run all examples."""
    print("Causal Intervention Models for User Retention - Examples")
    print("=" * 80)
    
    # Run examples
    analyzer1 = example_1_basic_analysis()
    analyzer2 = example_2_uplift_modeling()
    analyzer3 = example_3_user_segmentation()
    analyzer4 = example_4_ab_testing()
    analyzer5, results = example_5_comprehensive_analysis()
    custom_data = example_6_custom_scenarios()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("- causal_retention_analysis.png (comprehensive visualizations)")
    print("- ab_test_results.png (A/B testing results)")
    print("\nKey insights from the analysis:")
    print("1. Causal inference can identify true treatment effects")
    print("2. Uplift modeling helps personalize interventions")
    print("3. User segmentation enables targeted strategies")
    print("4. A/B testing simulation validates intervention effectiveness")
    print("5. Comprehensive analysis provides actionable insights")

if __name__ == "__main__":
    main()
