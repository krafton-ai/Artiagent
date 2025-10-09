"""
Simple test script to debug LEGION import issues.
Run this in the legion1.4.7 environment to check module availability.
"""

import os
import sys

print("=== LEGION Import Test ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")

# Check basic imports
print("\n=== Basic Imports ===")
try:
    import cv2
    print("✅ cv2 imported successfully")
except ImportError as e:
    print(f"❌ cv2 import failed: {e}")

try:
    import torch
    print("✅ torch imported successfully")
except ImportError as e:
    print(f"❌ torch import failed: {e}")

try:
    import bleach
    print("✅ bleach imported successfully")
except ImportError as e:
    print(f"❌ bleach import failed: {e}")

try:
    from transformers import AutoTokenizer
    print("✅ transformers.AutoTokenizer imported successfully")
except ImportError as e:
    print(f"❌ transformers.AutoTokenizer import failed: {e}")

# Check LEGION directory structure
print("\n=== LEGION Directory Structure ===")
legion_base = "/home/jhpark/LEGION"
if os.path.exists(legion_base):
    print(f"✅ LEGION base directory found: {legion_base}")
    
    # List contents
    for item in os.listdir(legion_base):
        item_path = os.path.join(legion_base, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
            
    # Check for specific directories
    expected_dirs = ['model', 'tools', 'eval', 'src']
    for dirname in expected_dirs:
        dir_path = os.path.join(legion_base, dirname)
        if os.path.exists(dir_path):
            print(f"✅ Found {dirname}/ directory")
        else:
            print(f"❌ Missing {dirname}/ directory")
else:
    print(f"❌ LEGION base directory not found: {legion_base}")

# Test LEGION imports with proper path setup
print("\n=== LEGION Imports Test ===")

# Add paths
legion_paths = [
    "/home/jhpark/LEGION",
    "/home/jhpark/LEGION/src", 
    "/home/jhpark/LEGION/model",
    "/home/jhpark/LEGION/tools",
    "/home/jhpark/LEGION/eval",
]

for path in legion_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added to Python path: {path}")

print("\nTrying LEGION imports...")

try:
    from model.Legion import LegionForCausalLM
    print("✅ model.Legion.LegionForCausalLM imported successfully")
except ImportError as e:
    print(f"❌ model.Legion.LegionForCausalLM import failed: {e}")

try:
    from model.llava import conversation as conversation_lib
    print("✅ model.llava.conversation imported successfully")
except ImportError as e:
    print(f"❌ model.llava.conversation import failed: {e}")

try:
    from model.llava.mm_utils import tokenizer_image_token
    print("✅ model.llava.mm_utils.tokenizer_image_token imported successfully") 
except ImportError as e:
    print(f"❌ model.llava.mm_utils.tokenizer_image_token import failed: {e}")

try:
    from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    print("✅ tools.utils constants imported successfully")
except ImportError as e:
    print(f"❌ tools.utils constants import failed: {e}")

try:
    from eval.utils import grounding_image_ecoder_preprocess
    print("✅ eval.utils.grounding_image_ecoder_preprocess imported successfully")
except ImportError as e:
    print(f"❌ eval.utils.grounding_image_ecoder_preprocess import failed: {e}")

print("\n=== Model Path Test ===")
model_path = "/data2/jhpark/LEGION/exp/Legion/final_model/global_step7030"
if os.path.exists(model_path):
    print(f"✅ LEGION model path found: {model_path}")
    print("Contents:")
    for item in os.listdir(model_path):
        print(f"  {item}")
else:
    print(f"❌ LEGION model path not found: {model_path}")

print("\n=== Test Complete ===")
