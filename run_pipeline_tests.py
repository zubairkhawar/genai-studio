#!/usr/bin/env python3
"""
Simple Pipeline Test Runner

Quick script to run video pipeline tests with the prompt: "Ocean waves crashing against rocky cliffs"
"""

import asyncio
import sys
import pathlib

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

async def main():
    """Run the pipeline tests"""
    try:
        from test_video_pipelines import VideoPipelineTester
        
        print("🚀 Starting Video Pipeline Tests")
        print("🎯 Test Prompt: 'Ocean waves crashing against rocky cliffs'")
        print("=" * 60)
        
        tester = VideoPipelineTester()
        await tester.run_all_tests()
        
        print("\n✅ All tests completed! Check the outputs/videos/pipeline_tests/ directory for results.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory.")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

