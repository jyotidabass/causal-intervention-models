"""
Test Script for Causal Intervention Models
==========================================

Simple test to verify that the system works correctly.
"""

import sys
import traceback
from causal_retention import CausalRetentionAnalyzer, UserRetentionDataGenerator

def test_data_generation():
    """Test data generation."""
    print("Testing data generation...")
    try:
        generator = UserRetentionDataGenerator(n_samples=1000)
        data = generator.generate_data()
        assert len(data) == 1000
        assert 'retention' in data.columns
        assert 'treatment' in data.columns
        print("✅ Data generation test passed")
        return True
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_analyzer_initialization():
    """Test analyzer initialization."""
    print("Testing analyzer initialization...")
    try:
        generator = UserRetentionDataGenerator(n_samples=1000)
        data = generator.generate_data()
        analyzer = CausalRetentionAnalyzer(data)
        assert analyzer.data is not None
        print("✅ Analyzer initialization test passed")
        return True
    except Exception as e:
        print(f"❌ Analyzer initialization test failed: {e}")
        return False

def test_user_segmentation():
    """Test user segmentation."""
    print("Testing user segmentation...")
    try:
        generator = UserRetentionDataGenerator(n_samples=1000)
        data = generator.generate_data()
        analyzer = CausalRetentionAnalyzer(data)
        segments = analyzer.create_user_segments()
        assert len(segments) == 1000
        assert len(set(segments)) == 5
        print("✅ User segmentation test passed")
        return True
    except Exception as e:
        print(f"❌ User segmentation test failed: {e}")
        return False

def test_intervention_recommendations():
    """Test intervention recommendations."""
    print("Testing intervention recommendations...")
    try:
        generator = UserRetentionDataGenerator(n_samples=1000)
        data = generator.generate_data()
        analyzer = CausalRetentionAnalyzer(data)
        analyzer.create_user_segments()
        recommendations = analyzer.generate_intervention_recommendations()
        assert len(recommendations) == 5
        assert all('interventions' in rec for rec in recommendations.values())
        print("✅ Intervention recommendations test passed")
        return True
    except Exception as e:
        print(f"❌ Intervention recommendations test failed: {e}")
        return False

def test_ab_testing():
    """Test A/B testing simulation."""
    print("Testing A/B testing simulation...")
    try:
        generator = UserRetentionDataGenerator(n_samples=1000)
        data = generator.generate_data()
        analyzer = CausalRetentionAnalyzer(data)
        ab_results = analyzer.run_ab_test_simulation(n_tests=5, sample_size=200)
        assert len(ab_results) == 5
        assert 'lift' in ab_results.columns
        assert 'p_value' in ab_results.columns
        print("✅ A/B testing simulation test passed")
        return True
    except Exception as e:
        print(f"❌ A/B testing simulation test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("RUNNING SYSTEM TESTS")
    print("=" * 50)
    
    tests = [
        test_data_generation,
        test_analyzer_initialization,
        test_user_segmentation,
        test_intervention_recommendations,
        test_ab_testing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            traceback.print_exc()
        print()
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
