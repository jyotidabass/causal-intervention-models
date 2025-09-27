"""
Causal Intervention Models for User Retention
============================================

A comprehensive framework for implementing causal inference methods to optimize 
user retention strategies using DoWhy and CausalML.

Author: AI Assistant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Causal inference libraries
try:
    import dowhy
    from dowhy import CausalModel
    from dowhy.utils import plot_causal_graph
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    print("Warning: DoWhy not available. Install with: pip install dowhy")

try:
    from causalml.inference.meta import LRSRegressor, XGBTRegressor
    from causalml.inference.tree import UpliftTreeClassifier
    from causalml.dataset import synthetic_data
    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False
    print("Warning: CausalML not available. Install with: pip install causalml")


class UserRetentionDataGenerator:
    """Generate synthetic user retention data with realistic causal relationships."""
    
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_data(self):
        """Generate synthetic user retention dataset."""
        print("Generating synthetic user retention data...")
        
        # User demographics and characteristics
        age = np.random.normal(35, 12, self.n_samples).clip(18, 80)
        income = np.random.lognormal(10, 0.5, self.n_samples)
        education = np.random.choice([1, 2, 3, 4, 5], self.n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
        gender = np.random.choice([0, 1], self.n_samples, p=[0.5, 0.5])
        
        # Device and platform features
        device_type = np.random.choice([0, 1, 2], self.n_samples, p=[0.6, 0.3, 0.1])  # mobile, desktop, tablet
        platform = np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3])  # iOS, Android
        
        # User engagement features (causal relationships)
        # Higher income -> more time spent
        time_spent = np.random.normal(20 + 0.01 * income, 10, self.n_samples).clip(0, 120)
        
        # Education -> more feature usage
        feature_usage = np.random.poisson(education * 2 + time_spent * 0.1, self.n_samples)
        
        # Age -> different app preferences
        app_preference = np.where(age < 30, 1, np.where(age < 50, 2, 3))
        
        # User acquisition channel
        acquisition_channel = np.random.choice([0, 1, 2, 3], self.n_samples, 
                                             p=[0.4, 0.3, 0.2, 0.1])  # organic, paid, referral, social
        
        # Session frequency (affects retention)
        session_frequency = np.random.poisson(3 + time_spent * 0.05, self.n_samples)
        
        # Treatment assignment (interventions)
        # Higher engagement users more likely to get premium features
        treatment_prob = 0.3 + 0.4 * (time_spent / time_spent.max()) + 0.2 * (feature_usage / feature_usage.max())
        treatment = np.random.binomial(1, treatment_prob.clip(0, 1))
        
        # Premium features intervention
        premium_features = np.random.binomial(1, 0.2 + 0.3 * treatment, self.n_samples)
        
        # Customer support interactions
        support_interactions = np.random.poisson(0.5 + 0.3 * (1 - treatment), self.n_samples)
        
        # Retention outcome (30-day retention)
        # Causal effects: treatment, premium_features, time_spent, session_frequency
        retention_logit = (
            -2.5 +  # baseline
            0.8 * treatment +  # treatment effect
            0.6 * premium_features +  # premium features effect
            0.05 * time_spent +  # engagement effect
            0.3 * session_frequency +  # frequency effect
            0.2 * feature_usage +  # feature usage effect
            -0.1 * support_interactions +  # support indicates problems
            0.1 * education +  # education effect
            -0.01 * age +  # age effect (slight)
            0.0001 * income +  # income effect (small)
            np.random.normal(0, 0.5, self.n_samples)  # random noise
        )
        
        retention_prob = 1 / (1 + np.exp(-retention_logit))
        retention = np.random.binomial(1, retention_prob)
        
        # Create DataFrame
        data = pd.DataFrame({
            'user_id': range(self.n_samples),
            'age': age,
            'income': income,
            'education': education,
            'gender': gender,
            'device_type': device_type,
            'platform': platform,
            'time_spent': time_spent,
            'feature_usage': feature_usage,
            'app_preference': app_preference,
            'acquisition_channel': acquisition_channel,
            'session_frequency': session_frequency,
            'treatment': treatment,
            'premium_features': premium_features,
            'support_interactions': support_interactions,
            'retention': retention,
            'retention_prob': retention_prob
        })
        
        print(f"Generated {len(data)} user records")
        print(f"Overall retention rate: {retention.mean():.3f}")
        print(f"Treatment group retention: {data[data.treatment==1].retention.mean():.3f}")
        print(f"Control group retention: {data[data.treatment==0].retention.mean():.3f}")
        
        return data


class CausalRetentionAnalyzer:
    """Main class for causal inference analysis of user retention."""
    
    def __init__(self, data=None):
        self.data = data
        self.causal_model = None
        self.estimated_effects = {}
        self.user_segments = None
        self.intervention_recommendations = None
        
    def load_data(self, data):
        """Load user retention data."""
        self.data = data
        print(f"Loaded data with {len(data)} records")
        
    def run_causal_analysis(self):
        """Run causal inference analysis using DoWhy."""
        if not DOWHY_AVAILABLE:
            print("DoWhy not available. Skipping causal analysis.")
            return None
            
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        print("\n=== Running Causal Analysis ===")
        
        # Define causal graph
        causal_graph = """
        digraph {
            age -> time_spent;
            income -> time_spent;
            education -> feature_usage;
            time_spent -> feature_usage;
            age -> app_preference;
            time_spent -> session_frequency;
            feature_usage -> session_frequency;
            treatment -> premium_features;
            treatment -> retention;
            premium_features -> retention;
            time_spent -> retention;
            session_frequency -> retention;
            feature_usage -> retention;
            support_interactions -> retention;
            education -> retention;
            income -> retention;
        }
        """
        
        # Create causal model
        self.causal_model = CausalModel(
            data=self.data,
            treatment='treatment',
            outcome='retention',
            graph=causal_graph
        )
        
        # Identify causal effect
        identified_estimand = self.causal_model.identify_effect(proceed_when_unidentifiable=True)
        print("Identified estimand:", identified_estimand)
        
        # Estimate causal effect
        causal_estimate = self.causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=True
        )
        
        print("Causal estimate:", causal_estimate)
        
        # Refute the estimate
        refute_results = self.causal_model.refute_estimate(
            identified_estimand, 
            causal_estimate,
            method_name="random_common_cause"
        )
        print("Refutation results:", refute_results)
        
        self.estimated_effects['treatment'] = {
            'estimate': causal_estimate.value,
            'confidence_interval': causal_estimate.get_confidence_intervals(),
            'p_value': causal_estimate.p_value
        }
        
        return causal_estimate
    
    def run_uplift_modeling(self):
        """Run uplift modeling using CausalML."""
        if not CAUSALML_AVAILABLE:
            print("CausalML not available. Skipping uplift modeling.")
            return None
            
        print("\n=== Running Uplift Modeling ===")
        
        # Prepare features for uplift modeling
        feature_cols = ['age', 'income', 'education', 'gender', 'device_type', 
                       'platform', 'time_spent', 'feature_usage', 'app_preference',
                       'acquisition_channel', 'session_frequency']
        
        X = self.data[feature_cols].values
        y = self.data['retention'].values
        treatment = self.data['treatment'].values
        
        # Split data
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, treatment, test_size=0.2, random_state=42, stratify=treatment
        )
        
        # Train uplift models
        uplift_models = {}
        
        # Linear Regression S-learner
        lr_s = LRSRegressor()
        lr_s.fit(X_train, y_train, t_train)
        uplift_lr_s = lr_s.predict(X_test, treatment=t_test) - lr_s.predict(X_test, treatment=1-t_test)
        
        # XGBoost T-learner
        xgb_t = XGBTRegressor()
        xgb_t.fit(X_train, y_train, t_train)
        uplift_xgb_t = xgb_t.predict(X_test, treatment=t_test) - xgb_t.predict(X_test, treatment=1-t_test)
        
        # Uplift Tree
        uplift_tree = UpliftTreeClassifier(max_depth=5, min_samples_leaf=200, min_samples_treatment=50)
        uplift_tree.fit(X_train, y_train, t_train)
        uplift_tree_pred = uplift_tree.predict(X_test)
        
        uplift_models = {
            'Linear_S_Learner': uplift_lr_s,
            'XGBoost_T_Learner': uplift_xgb_t,
            'Uplift_Tree': uplift_tree_pred
        }
        
        print("Uplift modeling completed")
        return uplift_models
    
    def create_user_segments(self):
        """Create user segments for targeted interventions."""
        print("\n=== Creating User Segments ===")
        
        # Features for clustering
        segment_features = ['age', 'income', 'time_spent', 'feature_usage', 
                           'session_frequency', 'education']
        
        X_segment = self.data[segment_features].copy()
        
        # Standardize features
        scaler = StandardScaler()
        X_segment_scaled = scaler.fit_transform(X_segment)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        segments = kmeans.fit_predict(X_segment_scaled)
        
        self.data['user_segment'] = segments
        
        # Analyze segments
        segment_analysis = self.data.groupby('user_segment').agg({
            'retention': ['mean', 'count'],
            'treatment': 'mean',
            'time_spent': 'mean',
            'feature_usage': 'mean',
            'session_frequency': 'mean',
            'income': 'mean',
            'age': 'mean'
        }).round(3)
        
        print("User segment analysis:")
        print(segment_analysis)
        
        self.user_segments = segment_analysis
        return segments
    
    def generate_intervention_recommendations(self):
        """Generate intervention recommendations based on causal analysis."""
        print("\n=== Generating Intervention Recommendations ===")
        
        if self.user_segments is None:
            self.create_user_segments()
        
        recommendations = {}
        
        for segment in range(5):
            segment_data = self.data[self.data['user_segment'] == segment]
            
            # Calculate segment characteristics
            retention_rate = segment_data['retention'].mean()
            avg_time_spent = segment_data['time_spent'].mean()
            avg_feature_usage = segment_data['feature_usage'].mean()
            avg_session_freq = segment_data['session_frequency'].mean()
            
            # Generate recommendations based on segment characteristics
            if retention_rate < 0.3:
                priority = "High"
                interventions = [
                    "Implement personalized onboarding flow",
                    "Send targeted re-engagement campaigns",
                    "Offer premium features trial",
                    "Provide proactive customer support"
                ]
            elif retention_rate < 0.5:
                priority = "Medium"
                interventions = [
                    "Optimize user engagement features",
                    "Implement gamification elements",
                    "Send feature usage tutorials",
                    "Offer loyalty rewards"
                ]
            else:
                priority = "Low"
                interventions = [
                    "Maintain current experience",
                    "Collect feedback for improvements",
                    "Offer referral incentives",
                    "Implement advanced features"
                ]
            
            # Segment-specific recommendations
            if avg_time_spent < 15:
                interventions.append("Improve app performance and loading times")
            if avg_feature_usage < 3:
                interventions.append("Increase feature discoverability")
            if avg_session_freq < 2:
                interventions.append("Implement push notifications for re-engagement")
            
            recommendations[segment] = {
                'priority': priority,
                'retention_rate': retention_rate,
                'interventions': interventions,
                'expected_impact': 'High' if priority == 'High' else 'Medium' if priority == 'Medium' else 'Low'
            }
        
        self.intervention_recommendations = recommendations
        
        print("Intervention recommendations generated for all segments")
        return recommendations
    
    def run_ab_test_simulation(self, n_tests=100, sample_size=1000):
        """Simulate A/B tests for different interventions."""
        print(f"\n=== Running A/B Test Simulation ({n_tests} tests) ===")
        
        ab_results = []
        
        for test_id in range(n_tests):
            # Randomly sample users for A/B test
            test_users = self.data.sample(n=sample_size, random_state=test_id)
            
            # Random assignment to treatment/control
            treatment_assignment = np.random.choice([0, 1], size=sample_size, p=[0.5, 0.5])
            
            # Simulate intervention effect (random effect size)
            effect_size = np.random.normal(0.05, 0.02)  # Small positive effect on average
            
            # Calculate retention with intervention effect
            baseline_retention = test_users['retention_prob'].values
            intervention_effect = effect_size * treatment_assignment
            retention_with_effect = np.minimum(1, baseline_retention + intervention_effect)
            
            # Simulate observed retention
            observed_retention = np.random.binomial(1, retention_with_effect)
            
            # Calculate test statistics
            control_retention = observed_retention[treatment_assignment == 0].mean()
            treatment_retention = observed_retention[treatment_assignment == 1].mean()
            lift = treatment_retention - control_retention
            
            # Statistical significance test
            from scipy.stats import chi2_contingency
            contingency_table = np.array([
                [observed_retention[treatment_assignment == 0].sum(), 
                 (treatment_assignment == 0).sum() - observed_retention[treatment_assignment == 0].sum()],
                [observed_retention[treatment_assignment == 1].sum(), 
                 (treatment_assignment == 1).sum() - observed_retention[treatment_assignment == 1].sum()]
            ])
            
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            
            ab_results.append({
                'test_id': test_id,
                'effect_size': effect_size,
                'control_retention': control_retention,
                'treatment_retention': treatment_retention,
                'lift': lift,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        ab_results_df = pd.DataFrame(ab_results)
        
        # Calculate power analysis
        significant_tests = ab_results_df['significant'].sum()
        power = significant_tests / n_tests
        
        print(f"A/B Test Simulation Results:")
        print(f"Total tests: {n_tests}")
        print(f"Significant tests: {significant_tests}")
        print(f"Statistical power: {power:.3f}")
        print(f"Average lift: {ab_results_df['lift'].mean():.4f}")
        print(f"Average p-value: {ab_results_df['p_value'].mean():.4f}")
        
        return ab_results_df
    
    def create_visualizations(self):
        """Create comprehensive visualizations for the analysis."""
        print("\n=== Creating Visualizations ===")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Causal Inference Analysis for User Retention', fontsize=16, fontweight='bold')
        
        # 1. Retention by treatment
        treatment_retention = self.data.groupby('treatment')['retention'].agg(['mean', 'count'])
        axes[0, 0].bar(['Control', 'Treatment'], treatment_retention['mean'])
        axes[0, 0].set_title('Retention Rate by Treatment Group')
        axes[0, 0].set_ylabel('Retention Rate')
        
        # 2. Time spent distribution
        axes[0, 1].hist(self.data['time_spent'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of Time Spent')
        axes[0, 1].set_xlabel('Time Spent (minutes)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Feature usage vs retention
        feature_retention = self.data.groupby('feature_usage')['retention'].mean()
        axes[0, 2].plot(feature_retention.index, feature_retention.values, marker='o')
        axes[0, 2].set_title('Feature Usage vs Retention')
        axes[0, 2].set_xlabel('Feature Usage Count')
        axes[0, 2].set_ylabel('Retention Rate')
        
        # 4. User segments retention
        if 'user_segment' in self.data.columns:
            segment_retention = self.data.groupby('user_segment')['retention'].mean()
            axes[1, 0].bar(segment_retention.index, segment_retention.values)
            axes[1, 0].set_title('Retention by User Segment')
            axes[1, 0].set_xlabel('User Segment')
            axes[1, 0].set_ylabel('Retention Rate')
        
        # 5. Correlation heatmap
        corr_features = ['age', 'income', 'time_spent', 'feature_usage', 
                        'session_frequency', 'retention']
        corr_matrix = self.data[corr_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation Matrix')
        
        # 6. Treatment effect by segments
        if 'user_segment' in self.data.columns:
            treatment_effect_by_segment = self.data.groupby(['user_segment', 'treatment'])['retention'].mean().unstack()
            treatment_effect_by_segment.plot(kind='bar', ax=axes[1, 2])
            axes[1, 2].set_title('Treatment Effect by User Segment')
            axes[1, 2].set_xlabel('User Segment')
            axes[1, 2].set_ylabel('Retention Rate')
            axes[1, 2].legend(['Control', 'Treatment'])
        
        # 7. Age distribution by retention
        retained = self.data[self.data['retention'] == 1]['age']
        not_retained = self.data[self.data['retention'] == 0]['age']
        axes[2, 0].hist([retained, not_retained], bins=20, alpha=0.7, 
                       label=['Retained', 'Not Retained'])
        axes[2, 0].set_title('Age Distribution by Retention')
        axes[2, 0].set_xlabel('Age')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].legend()
        
        # 8. Income vs retention
        income_retention = self.data.groupby(pd.cut(self.data['income'], bins=5))['retention'].mean()
        axes[2, 1].bar(range(len(income_retention)), income_retention.values)
        axes[2, 1].set_title('Retention by Income Quintile')
        axes[2, 1].set_xlabel('Income Quintile')
        axes[2, 1].set_ylabel('Retention Rate')
        
        # 9. Session frequency vs retention
        freq_retention = self.data.groupby('session_frequency')['retention'].mean()
        axes[2, 2].plot(freq_retention.index, freq_retention.values, marker='o')
        axes[2, 2].set_title('Session Frequency vs Retention')
        axes[2, 2].set_xlabel('Session Frequency')
        axes[2, 2].set_ylabel('Retention Rate')
        
        plt.tight_layout()
        plt.savefig('causal_retention_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'causal_retention_analysis.png'")
    
    def run_complete_analysis(self):
        """Run the complete causal inference analysis pipeline."""
        print("=== Starting Complete Causal Inference Analysis ===")
        
        # Generate data if not provided
        if self.data is None:
            generator = UserRetentionDataGenerator()
            self.data = generator.generate_data()
        
        # Run all analysis components
        results = {}
        
        # Causal analysis
        causal_results = self.run_causal_analysis()
        results['causal_analysis'] = causal_results
        
        # Uplift modeling
        uplift_results = self.run_uplift_modeling()
        results['uplift_modeling'] = uplift_results
        
        # User segmentation
        segments = self.create_user_segments()
        results['user_segments'] = segments
        
        # Intervention recommendations
        recommendations = self.generate_intervention_recommendations()
        results['intervention_recommendations'] = recommendations
        
        # A/B test simulation
        ab_results = self.run_ab_test_simulation()
        results['ab_test_results'] = ab_results
        
        # Visualizations
        self.create_visualizations()
        
        print("\n=== Analysis Complete ===")
        return results
    
    def get_summary_report(self):
        """Generate a summary report of all analysis results."""
        if self.data is None:
            return "No analysis has been run yet."
        
        report = []
        report.append("=== CAUSAL INFERENCE ANALYSIS SUMMARY ===\n")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append(f"Total users: {len(self.data):,}")
        report.append(f"Overall retention rate: {self.data['retention'].mean():.3f}")
        report.append(f"Treatment group size: {self.data['treatment'].sum():,}")
        report.append(f"Control group size: {(self.data['treatment'] == 0).sum():,}")
        report.append("")
        
        # Treatment effects
        if self.estimated_effects:
            report.append("CAUSAL EFFECTS:")
            for treatment, effect in self.estimated_effects.items():
                report.append(f"{treatment.title()} effect: {effect['estimate']:.4f}")
                report.append(f"P-value: {effect['p_value']:.4f}")
            report.append("")
        
        # User segments
        if self.user_segments is not None:
            report.append("USER SEGMENT ANALYSIS:")
            for segment in range(5):
                segment_data = self.data[self.data['user_segment'] == segment]
                report.append(f"Segment {segment}: {len(segment_data):,} users, "
                            f"retention rate: {segment_data['retention'].mean():.3f}")
            report.append("")
        
        # Intervention recommendations
        if self.intervention_recommendations:
            report.append("INTERVENTION RECOMMENDATIONS:")
            for segment, rec in self.intervention_recommendations.items():
                report.append(f"Segment {segment} ({rec['priority']} priority):")
                for intervention in rec['interventions'][:3]:  # Top 3 interventions
                    report.append(f"  - {intervention}")
                report.append("")
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CausalRetentionAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Print summary report
    print(analyzer.get_summary_report())
