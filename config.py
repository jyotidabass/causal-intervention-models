"""
Configuration file for Causal Intervention Models
================================================

This file contains all configurable parameters for the causal retention analysis.
Modify these settings to customize the analysis for your specific use case.
"""

# Data Generation Parameters
DATA_CONFIG = {
    'n_samples': 10000,
    'random_state': 42,
    
    # User demographics
    'age_mean': 35,
    'age_std': 12,
    'age_min': 18,
    'age_max': 80,
    
    'income_log_mean': 10,
    'income_log_std': 0.5,
    
    'education_probs': [0.1, 0.2, 0.3, 0.3, 0.1],  # education levels 1-5
    'gender_probs': [0.5, 0.5],  # gender 0, 1
    
    # Device and platform
    'device_probs': [0.6, 0.3, 0.1],  # mobile, desktop, tablet
    'platform_probs': [0.7, 0.3],  # iOS, Android
    
    # Acquisition channels
    'acquisition_probs': [0.4, 0.3, 0.2, 0.1],  # organic, paid, referral, social
    
    # Treatment assignment
    'treatment_base_prob': 0.3,
    'treatment_engagement_weight': 0.4,
    'treatment_feature_weight': 0.2,
    
    # Retention model coefficients
    'retention_baseline': -2.5,
    'treatment_effect': 0.8,
    'premium_effect': 0.6,
    'time_spent_effect': 0.05,
    'session_freq_effect': 0.3,
    'feature_usage_effect': 0.2,
    'support_effect': -0.1,
    'education_effect': 0.1,
    'age_effect': -0.01,
    'income_effect': 0.0001,
    'noise_std': 0.5
}

# Analysis Parameters
ANALYSIS_CONFIG = {
    # Causal analysis
    'causal_methods': [
        'backdoor.linear_regression',
        'backdoor.propensity_score_stratification',
        'backdoor.propensity_score_matching'
    ],
    
    # Uplift modeling
    'uplift_models': [
        'LRSRegressor',      # Linear Regression S-learner
        'XGBTRegressor',     # XGBoost T-learner
        'UpliftTreeClassifier'  # Uplift Tree
    ],
    
    # User segmentation
    'n_segments': 5,
    'segment_features': [
        'age', 'income', 'time_spent', 'feature_usage', 
        'session_frequency', 'education'
    ],
    
    # A/B testing simulation
    'ab_test_n_tests': 100,
    'ab_test_sample_size': 1000,
    'ab_test_significance_level': 0.05,
    
    # Train/test split
    'test_size': 0.2,
    'validation_size': 0.1
}

# Intervention Recommendations
INTERVENTION_CONFIG = {
    'high_priority_threshold': 0.3,
    'medium_priority_threshold': 0.5,
    
    'interventions_by_priority': {
        'high': [
            "Implement personalized onboarding flow",
            "Send targeted re-engagement campaigns",
            "Offer premium features trial",
            "Provide proactive customer support",
            "Implement exit-intent popups",
            "Send win-back email series"
        ],
        'medium': [
            "Optimize user engagement features",
            "Implement gamification elements",
            "Send feature usage tutorials",
            "Offer loyalty rewards",
            "Create user onboarding checklist",
            "Implement push notifications"
        ],
        'low': [
            "Maintain current experience",
            "Collect feedback for improvements",
            "Offer referral incentives",
            "Implement advanced features",
            "Create user community",
            "Provide educational content"
        ]
    },
    
    'segment_specific_interventions': {
        'low_engagement': [
            "Improve app performance and loading times",
            "Simplify user interface",
            "Provide quick wins and easy tasks"
        ],
        'low_feature_usage': [
            "Increase feature discoverability",
            "Create feature tutorials",
            "Implement progressive disclosure"
        ],
        'low_frequency': [
            "Implement push notifications for re-engagement",
            "Send personalized content recommendations",
            "Create daily/weekly habits"
        ],
        'high_value': [
            "Provide VIP support",
            "Offer exclusive features",
            "Implement priority features"
        ]
    }
}

# Visualization Settings
VISUALIZATION_CONFIG = {
    'figure_size': (18, 15),
    'dpi': 300,
    'style': 'default',
    'palette': 'husl',
    
    'colors': {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'success': '#F18F01',
        'warning': '#C73E1D',
        'info': '#6C757D'
    },
    
    'plot_settings': {
        'alpha': 0.7,
        'edgecolor': 'black',
        'linewidth': 2,
        'markersize': 6
    }
}

# Model Parameters
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    
    'kmeans': {
        'n_clusters': 5,
        'random_state': 42,
        'n_init': 10,
        'max_iter': 300
    },
    
    'uplift_tree': {
        'max_depth': 5,
        'min_samples_leaf': 200,
        'min_samples_treatment': 50,
        'random_state': 42
    }
}

# Output Settings
OUTPUT_CONFIG = {
    'save_plots': True,
    'plot_format': 'png',
    'save_data': False,
    'data_format': 'csv',
    
    'output_files': {
        'main_analysis': 'causal_retention_analysis.png',
        'ab_test_results': 'ab_test_results.png',
        'segment_analysis': 'user_segments_analysis.png',
        'uplift_results': 'uplift_modeling_results.png'
    },
    
    'report_format': 'text',  # 'text', 'html', 'json'
    'include_detailed_results': True
}

# Feature Engineering
FEATURE_CONFIG = {
    'numerical_features': [
        'age', 'income', 'time_spent', 'feature_usage', 
        'session_frequency', 'education'
    ],
    
    'categorical_features': [
        'gender', 'device_type', 'platform', 'app_preference',
        'acquisition_channel'
    ],
    
    'target_variable': 'retention',
    'treatment_variable': 'treatment',
    
    'feature_scaling': True,
    'feature_selection': True,
    'max_features': 20
}

# Validation Settings
VALIDATION_CONFIG = {
    'cross_validation_folds': 5,
    'stratified': True,
    'shuffle': True,
    'random_state': 42,
    
    'metrics': [
        'accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'uplift_at_k'
    ],
    
    'uplift_metrics': {
        'qini_score': True,
        'uplift_at_k': [10, 20, 30],  # percentiles
        'auuc': True  # Area Under Uplift Curve
    }
}
