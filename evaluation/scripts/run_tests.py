#!/usr/bin/env python3
"""
Evaluation Test Runner

Quick test runner for the ReUpload dataset evaluation
"""

import subprocess
import sys
from pathlib import Path

def run_structure_test():
    """Run the structure exploration test"""
    print("🔍 Testing ReUpload dataset structure...")
    
    # Ask user for data path
    data_path = input("Enter path to ReUpload dataset: ")
    if not data_path:
        print("❌ No path provided")
        return
    
    # Run structure test
    script_path = Path(__file__).parent / "test_reupload_structure.py"
    cmd = [sys.executable, str(script_path), "--data", data_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"❌ Failed to run structure test: {e}")

def run_evaluation_test():
    """Run a small evaluation test"""
    print("🧪 Running ReUpload evaluation test...")
    
    # Ask user for data path
    data_path = input("Enter path to ReUpload dataset: ")
    if not data_path:
        print("❌ No path provided")
        return
    
    # Run evaluation with max 2 samples
    script_path = Path(__file__).parent / "reupload_evaluation.py"
    cmd = [sys.executable, str(script_path), "--data", data_path, "--max-samples", "2", "--verbose"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"❌ Failed to run evaluation test: {e}")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*50)
        print("REUPLOAD EVALUATION TEST RUNNER")
        print("="*50)
        print("1. Test dataset structure")
        print("2. Run small evaluation test (2 samples)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == "1":
            run_structure_test()
        elif choice == "2":
            run_evaluation_test()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
