#!/usr/bin/env python3
"""
Quick runner for Ultimate Video Generator tests

This script runs comprehensive tests of the Ultimate Video Generator
with SD + AnimateDiff using multiple prompts and balanced settings.
"""

import asyncio
import sys
import pathlib

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

async def main():
    """Run the Ultimate Video Generator tests"""
    try:
        from test_ultimate_video_generator import UltimateVideoGeneratorTester
        
        print("🚀 Starting Ultimate Video Generator Tests")
        print("🎯 Testing SD + AnimateDiff with multiple prompts")
        print("⚖️ Using balanced settings for optimal quality/performance")
        print("=" * 70)
        
        tester = UltimateVideoGeneratorTester()
        await tester.run_comprehensive_tests()
        
        print("\n✅ All Ultimate Video Generator tests completed!")
        print("📁 Check the outputs/videos/ultimate_tests/ directory for results.")
        print("📄 Check ultimate_video_test_results.log for detailed logs.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory.")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

