#!/usr/bin/env python3
"""
Test runner for all LLM judge tests.
Runs comprehensive test suites for all judges in the AgentEval framework.
"""

import sys
import os
import unittest
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def discover_and_run_tests(test_pattern='test_*.py', verbosity=2):
    """Discover and run all judge tests."""
    
    # Get the judges test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"🧪 Running LLM Judge Tests from: {test_dir}")
    print("=" * 70)
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=test_pattern)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ ALL JUDGE TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    print("=" * 70)
    
    return success


def run_specific_test_file(test_file, verbosity=2):
    """Run tests from a specific test file."""
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(test_dir, test_file)
    
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        return False
    
    print(f"🧪 Running tests from: {test_file}")
    print("=" * 50)
    
    # Load and run specific test file
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_file.replace('.py', ''))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    return success


def list_available_tests():
    """List all available test files."""
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    
    print("📋 Available Judge Test Files:")
    print("=" * 40)
    for test_file in sorted(test_files):
        print(f"  - {test_file}")
    
    return test_files


def validate_judge_setup():
    """Validate that judges are properly set up and can be imported."""
    
    print("🔍 Validating Judge Setup...")
    print("-" * 30)
    
    validation_errors = []
    
    try:
        # Test judge registry
        from agent_eval.judges.base import JudgeRegistry
        JudgeRegistry._build_maps()
        print("✅ Judge registry loaded successfully")
        
        # Test core judges
        core_judges = [
            'factuality', 'fluency', 'relevance', 'helpfulness', 'safety', 'creativity'
        ]
        
        for judge_name in core_judges:
            judge_class = JudgeRegistry.get_by_name(judge_name)
            if judge_class:
                print(f"✅ Core judge '{judge_name}' found")
            else:
                validation_errors.append(f"Core judge '{judge_name}' not found")
        
        # Test new judges  
        new_judges = [
            'coherence', 'completeness', 'bias', 'healthcare', 
            'technical_accuracy', 'legal', 'empathy', 'educational',
            'aml_compliance', 'financial_expertise'
        ]
        
        for judge_name in new_judges:
            judge_class = JudgeRegistry.get_by_name(judge_name)
            if judge_class:
                print(f"✅ New judge '{judge_name}' found")
            else:
                validation_errors.append(f"New judge '{judge_name}' not found")
        
        # Test intelligent selector
        from agent_eval.core.intelligent_selector import IntelligentSelector
        print("✅ Intelligent selector imported successfully")
        
    except Exception as e:
        validation_errors.append(f"Import error: {str(e)}")
    
    print("-" * 30)
    
    if validation_errors:
        print("❌ Validation Issues Found:")
        for error in validation_errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ All judges validated successfully!")
        return True


def main():
    """Main test runner function."""
    
    parser = argparse.ArgumentParser(description='Run LLM Judge Tests for AgentEval')
    parser.add_argument('--test-file', '-f', help='Run specific test file')
    parser.add_argument('--list', '-l', action='store_true', help='List available test files')
    parser.add_argument('--validate', '-v', action='store_true', help='Validate judge setup')
    parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2], help='Test verbosity level')
    parser.add_argument('--pattern', default='test_*.py', help='Test file pattern (default: test_*.py)')
    
    args = parser.parse_args()
    
    # Handle different command options
    if args.list:
        list_available_tests()
        return
    
    if args.validate:
        if not validate_judge_setup():
            sys.exit(1)
        return
    
    # Validate setup before running tests
    if not validate_judge_setup():
        print("\n❌ Judge setup validation failed. Please fix issues before running tests.")
        sys.exit(1)
    
    print()  # Add spacing after validation
    
    # Run specific test file or all tests
    if args.test_file:
        success = run_specific_test_file(args.test_file, args.verbosity)
    else:
        success = discover_and_run_tests(args.pattern, args.verbosity)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()