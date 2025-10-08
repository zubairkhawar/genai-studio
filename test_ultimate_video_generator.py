#!/usr/bin/env python3
"""
Ultimate Video Generator Testing Script

Comprehensive testing of the Ultimate Video Generator with SD + AnimateDiff
using multiple prompts and balanced settings.

Tests:
1. Multiple diverse prompts to test different scenarios
2. Balanced preset settings for optimal quality/performance
3. AnimateDiff parameter variations
4. Performance benchmarking
5. Quality assessment

Usage: python test_ultimate_video_generator.py
"""

import asyncio
import logging
import time
import sys
import pathlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import random

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).parent / "backend"))

from backend.utils.gpu_detector import GPUDetector
from backend.models.ultimate_video_generator import UltimateVideoGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ultimate_video_test_results.log')
    ]
)
logger = logging.getLogger(__name__)

class UltimateVideoGeneratorTester:
    """Comprehensive testing framework for Ultimate Video Generator"""
    
    def __init__(self):
        gpu_detector = GPUDetector()
        self.gpu_info = gpu_detector.gpu_info
        self.generator = None
        self.results = {}
        self.start_time = None
        
        # Setup output directory
        project_root = pathlib.Path(__file__).parent
        self.outputs_dir = project_root / "outputs" / "videos" / "ultimate_tests"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Test prompts covering different scenarios
        self.test_prompts = [
            {
                "prompt": "Ocean waves crashing against rocky cliffs at sunset, cinematic lighting, dramatic clouds",
                "negative_prompt": "blurry, low quality, distorted, artifacts",
                "category": "nature",
                "description": "Natural scene with motion and lighting"
            },
            {
                "prompt": "A majestic eagle soaring through mountain peaks, golden hour lighting, photorealistic",
                "negative_prompt": "cartoon, anime, low quality, blurry",
                "category": "wildlife",
                "description": "Wildlife with dynamic movement"
            },
            {
                "prompt": "Steam rising from a hot cup of coffee on a wooden table, morning light, cozy atmosphere",
                "negative_prompt": "dark, gloomy, low quality, distorted",
                "category": "still_life",
                "description": "Subtle motion in still life"
            },
            {
                "prompt": "A dancer performing ballet moves in a grand theater, stage lighting, elegant movement",
                "negative_prompt": "blurry, low quality, amateur, distorted",
                "category": "human_motion",
                "description": "Complex human movement"
            },
            {
                "prompt": "Fireworks exploding in the night sky over a city skyline, colorful sparks, celebration",
                "negative_prompt": "daytime, low quality, blurry, dark",
                "category": "celebration",
                "description": "Dynamic light effects and motion"
            },
            {
                "prompt": "A cat playing with a ball of yarn, soft fur, warm indoor lighting, playful movement",
                "negative_prompt": "aggressive, scary, low quality, blurry",
                "category": "pets",
                "description": "Cute animal motion"
            }
        ]
        
        # AnimateDiff parameter variations for testing
        self.ad_variations = [
            {
                "name": "conservative",
                "params": {"frames": 8, "fps": 8, "motion_scale": 1.0, "steps": 20},
                "description": "Conservative motion, fast generation"
            },
            {
                "name": "balanced",
                "params": {"frames": 16, "fps": 8, "motion_scale": 1.4, "steps": 30},
                "description": "Balanced motion and quality"
            },
            {
                "name": "dynamic",
                "params": {"frames": 24, "fps": 8, "motion_scale": 1.8, "steps": 35},
                "description": "High motion, high quality"
            }
        ]
        
        logger.info(f"🚀 Initialized Ultimate Video Generator Tester")
        logger.info(f"📱 Device: {self.gpu_info['device']}")
        logger.info(f"💾 Output Directory: {self.outputs_dir}")
        logger.info(f"🎯 Test Prompts: {len(self.test_prompts)}")
        logger.info(f"⚙️ AnimateDiff Variations: {len(self.ad_variations)}")

    async def setup_generator(self) -> bool:
        """Initialize the Ultimate Video Generator"""
        try:
            logger.info("🔄 Initializing Ultimate Video Generator...")
            
            self.generator = UltimateVideoGenerator(self.gpu_info)
            
            # Load all models
            models_loaded = await self.generator.load_models()
            if not models_loaded:
                logger.error("❌ Failed to load Ultimate Video Generator models")
                return False
            
            logger.info("✅ Ultimate Video Generator initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Ultimate Video Generator: {e}")
            return False

    async def test_single_generation(
        self, 
        prompt_data: Dict[str, Any], 
        preset: str, 
        ad_variation: Dict[str, Any],
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Test a single video generation with specific parameters"""
        
        test_name = f"{prompt_data['category']}_{preset}_{ad_variation['name']}"
        logger.info(f"🧪 Testing: {test_name}")
        logger.info(f"  📝 Prompt: {prompt_data['prompt'][:60]}...")
        logger.info(f"  ⚙️ Preset: {preset}")
        logger.info(f"  🎬 AD Variation: {ad_variation['name']}")
        
        start_time = time.time()
        result = {
            "test_name": test_name,
            "prompt_data": prompt_data,
            "preset": preset,
            "ad_variation": ad_variation,
            "seed": seed,
            "success": False,
            "duration": 0,
            "output_path": None,
            "error": None,
            "progress_stages": []
        }
        
        def progress_callback(progress: int, message: str):
            result["progress_stages"].append({
                "progress": progress,
                "message": message,
                "timestamp": time.time()
            })
            logger.info(f"    📊 {progress}%: {message}")
        
        try:
            # Generate video
            output_path = await self.generator.generate(
                prompt=prompt_data["prompt"],
                preset=preset,
                negative_prompt=prompt_data.get("negative_prompt"),
                ad_overrides=ad_variation["params"],
                seed=seed,
                progress_callback=progress_callback
            )
            
            result["success"] = True
            result["output_path"] = output_path
            result["duration"] = time.time() - start_time
            
            # Copy to test outputs with descriptive name
            import shutil
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            test_output = self.outputs_dir / f"{test_name}_{timestamp}.mp4"
            shutil.copy2(output_path, test_output)
            result["test_output_path"] = str(test_output)
            
            logger.info(f"  ✅ {test_name} completed successfully in {result['duration']:.2f}s")
            logger.info(f"  📁 Output saved to: {test_output}")
            
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            logger.error(f"  ❌ {test_name} failed: {e}")
        
        return result

    async def test_preset_comparison(self) -> Dict[str, Any]:
        """Test all presets with a single prompt to compare performance"""
        logger.info("🔬 Testing Preset Comparison")
        
        # Use a single representative prompt
        test_prompt = self.test_prompts[0]  # Ocean waves
        ad_variation = self.ad_variations[1]  # Balanced
        
        presets = ["ultra-fast", "balanced", "high-quality"]
        results = {}
        
        for preset in presets:
            logger.info(f"\n📊 Testing Preset: {preset}")
            result = await self.test_single_generation(
                test_prompt, preset, ad_variation, seed=42
            )
            results[preset] = result
        
        return {
            "test_type": "preset_comparison",
            "test_prompt": test_prompt,
            "ad_variation": ad_variation,
            "results": results
        }

    async def test_ad_variations(self) -> Dict[str, Any]:
        """Test different AnimateDiff parameter variations"""
        logger.info("🎬 Testing AnimateDiff Variations")
        
        # Use a single representative prompt and preset
        test_prompt = self.test_prompts[1]  # Eagle
        preset = "balanced"
        
        results = {}
        
        for ad_variation in self.ad_variations:
            logger.info(f"\n⚙️ Testing AD Variation: {ad_variation['name']}")
            result = await self.test_single_generation(
                test_prompt, preset, ad_variation, seed=123
            )
            results[ad_variation["name"]] = result
        
        return {
            "test_type": "ad_variations",
            "test_prompt": test_prompt,
            "preset": preset,
            "results": results
        }

    async def test_prompt_diversity(self) -> Dict[str, Any]:
        """Test diverse prompts with balanced settings"""
        logger.info("🎨 Testing Prompt Diversity")
        
        preset = "balanced"
        ad_variation = self.ad_variations[1]  # Balanced
        results = {}
        
        for i, prompt_data in enumerate(self.test_prompts):
            logger.info(f"\n🎯 Testing Prompt {i+1}/{len(self.test_prompts)}: {prompt_data['category']}")
            result = await self.test_single_generation(
                prompt_data, preset, ad_variation, seed=100 + i
            )
            results[prompt_data["category"]] = result
        
        return {
            "test_type": "prompt_diversity",
            "preset": preset,
            "ad_variation": ad_variation,
            "results": results
        }

    async def test_balanced_optimization(self) -> Dict[str, Any]:
        """Test optimized balanced settings for different scenarios"""
        logger.info("⚖️ Testing Balanced Optimization")
        
        # Custom balanced settings for different scenarios
        optimized_tests = [
            {
                "name": "nature_scene",
                "prompt": self.test_prompts[0],  # Ocean waves
                "ad_params": {"frames": 16, "fps": 8, "motion_scale": 1.2, "steps": 25},
                "description": "Optimized for natural motion"
            },
            {
                "name": "human_motion",
                "prompt": self.test_prompts[3],  # Dancer
                "ad_params": {"frames": 20, "fps": 8, "motion_scale": 1.6, "steps": 30},
                "description": "Optimized for complex human movement"
            },
            {
                "name": "subtle_motion",
                "prompt": self.test_prompts[2],  # Coffee steam
                "ad_params": {"frames": 12, "fps": 8, "motion_scale": 0.8, "steps": 20},
                "description": "Optimized for subtle motion"
            }
        ]
        
        results = {}
        
        for test in optimized_tests:
            logger.info(f"\n🔧 Testing Optimized: {test['name']}")
            ad_variation = {
                "name": test["name"],
                "params": test["ad_params"],
                "description": test["description"]
            }
            
            result = await self.test_single_generation(
                test["prompt"], "balanced", ad_variation, seed=200
            )
            results[test["name"]] = result
        
        return {
            "test_type": "balanced_optimization",
            "results": results
        }

    async def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        self.start_time = time.time()
        logger.info("🚀 Starting Comprehensive Ultimate Video Generator Testing")
        logger.info("=" * 80)
        
        # Setup generator
        if not await self.setup_generator():
            logger.error("❌ Failed to setup generator. Exiting.")
            return
        
        # Run all test suites
        test_suites = [
            ("Preset Comparison", self.test_preset_comparison),
            ("AnimateDiff Variations", self.test_ad_variations),
            ("Prompt Diversity", self.test_prompt_diversity),
            ("Balanced Optimization", self.test_balanced_optimization),
        ]
        
        for suite_name, suite_method in test_suites:
            logger.info(f"\n{'='*20} {suite_name} {'='*20}")
            try:
                result = await suite_method()
                self.results[suite_name] = result
            except Exception as e:
                logger.error(f"❌ {suite_name} failed with exception: {e}")
                self.results[suite_name] = {
                    "test_type": suite_name.lower().replace(" ", "_"),
                    "success": False,
                    "error": str(e),
                    "results": {}
                }
        
        # Generate comprehensive report
        await self.generate_comprehensive_report()

    async def generate_comprehensive_report(self):
        """Generate a comprehensive test report"""
        total_duration = time.time() - self.start_time
        
        logger.info("\n" + "="*80)
        logger.info("📊 ULTIMATE VIDEO GENERATOR TESTING REPORT")
        logger.info("="*80)
        
        # Overall statistics
        total_tests = 0
        successful_tests = 0
        
        for suite_name, suite_result in self.results.items():
            if isinstance(suite_result, dict) and "results" in suite_result:
                suite_tests = len(suite_result["results"])
                suite_successful = sum(1 for r in suite_result["results"].values() if r.get("success", False))
                total_tests += suite_tests
                successful_tests += suite_successful
        
        logger.info(f"⏱️  Total Test Duration: {total_duration:.2f} seconds")
        logger.info(f"✅ Successful Tests: {successful_tests}/{total_tests}")
        logger.info(f"❌ Failed Tests: {total_tests - successful_tests}/{total_tests}")
        logger.info(f"📁 Output Directory: {self.outputs_dir}")
        
        # Detailed results by test suite
        for suite_name, suite_result in self.results.items():
            logger.info(f"\n📋 {suite_name} Results:")
            logger.info("-" * 60)
            
            if isinstance(suite_result, dict) and "results" in suite_result:
                for test_name, test_result in suite_result["results"].items():
                    status = "✅ PASS" if test_result.get("success", False) else "❌ FAIL"
                    duration = test_result.get("duration", 0)
                    output_path = test_result.get("test_output_path", "N/A")
                    
                    logger.info(f"{status} {test_name}")
                    logger.info(f"    Duration: {duration:.2f}s")
                    logger.info(f"    Output: {output_path}")
                    
                    if test_result.get("error"):
                        logger.info(f"    Error: {test_result['error']}")
                    
                    # Show progress stages for successful tests
                    if test_result.get("success") and test_result.get("progress_stages"):
                        logger.info("    Progress Stages:")
                        for stage in test_result["progress_stages"][::2]:  # Show every other stage
                            logger.info(f"      {stage['progress']}%: {stage['message']}")
                    
                    logger.info("")
        
        # Performance analysis
        logger.info("\n📈 Performance Analysis:")
        logger.info("-" * 40)
        
        all_durations = []
        for suite_result in self.results.values():
            if isinstance(suite_result, dict) and "results" in suite_result:
                for test_result in suite_result["results"].values():
                    if test_result.get("success") and test_result.get("duration"):
                        all_durations.append(test_result["duration"])
        
        if all_durations:
            avg_duration = sum(all_durations) / len(all_durations)
            min_duration = min(all_durations)
            max_duration = max(all_durations)
            
            logger.info(f"Average Generation Time: {avg_duration:.2f}s")
            logger.info(f"Fastest Generation: {min_duration:.2f}s")
            logger.info(f"Slowest Generation: {max_duration:.2f}s")
        
        # Save detailed results to JSON
        results_file = self.outputs_dir / "comprehensive_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "device": self.gpu_info["device"],
                "test_prompts": self.test_prompts,
                "ad_variations": self.ad_variations,
                "results": self.results
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
        logger.info("="*80)

async def main():
    """Main function to run the comprehensive tests"""
    tester = UltimateVideoGeneratorTester()
    await tester.run_comprehensive_tests()

if __name__ == "__main__":
    asyncio.run(main())

