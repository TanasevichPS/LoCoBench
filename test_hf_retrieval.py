#!/usr/bin/env python3
"""
Test script for Hugging Face models with retrieval pipeline
Tests small code models like Qwen2.5-Coder-1.5B or DeepSeek-Coder-1.3B
"""

import asyncio
import json
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from locobench.core.config import Config
from locobench.evaluation.evaluator import Evaluator
from locobench.generation.synthetic_generator import MultiLLMGenerator
from locobench.utils.llm_parsing import parse_llm_response

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_hf_model_generation():
    """Test Hugging Face model generation"""
    console.print(Panel.fit("🧪 Testing Hugging Face Model Generation", style="bold blue"))
    
    # Load config
    config = Config.from_yaml("config.yaml")
    
    # Initialize generator
    generator = MultiLLMGenerator(config)
    
    # Test prompt
    test_prompt = """Write a Python function that calculates the factorial of a number.
Return the response as JSON with this format:
```json
{
    "files": {
        "factorial.py": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n - 1)"
    }
}
```
"""
    
    # Test with a small Hugging Face code model
    # Use Qwen2.5-Coder-1.5B or DeepSeek-Coder-1.3B (small models)
    hf_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    try:
        console.print(f"🤖 Testing model: {hf_model}")
        console.print("⏳ Generating response (this may take a while on first load)...")
        
        response = await generator.generate_with_huggingface(hf_model, test_prompt)
        
        console.print(f"✅ Response received: {len(response)} characters")
        console.print(f"\n📝 Response preview:\n{response[:500]}...")
        
        # Test parsing
        parsed = parse_llm_response(response, expected_language='python')
        if parsed:
            console.print(f"✅ Parsing successful: {len(parsed)} files extracted")
            for filename, content in parsed.items():
                console.print(f"  📄 {filename}: {len(content)} chars")
        else:
            console.print("⚠️ Parsing failed - response may need different format")
            
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        import traceback
        console.print(traceback.format_exc())


async def test_retrieval_pipeline():
    """Test full retrieval pipeline with Hugging Face model"""
    console.print(Panel.fit("🔍 Testing Retrieval Pipeline with Hugging Face", style="bold green"))
    
    # Load config
    config = Config.from_yaml("config.yaml")
    
    # Enable retrieval
    config.retrieval.enabled = True
    config.retrieval.method = "embedding"
    config.retrieval.model_name = "all-MiniLM-L6-v2"  # Small embedding model
    
    # Create a simple test scenario
    test_context = {
        "utils.py": """
def calculate_factorial(n):
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
""",
        "main.py": """
import utils

def main():
    print(utils.calculate_factorial(5))
    print(utils.calculate_fibonacci(10))

if __name__ == "__main__":
    main()
"""
    }
    
    task_prompt = "Add a function to calculate the power of a number"
    
    # Test retrieval
    from locobench.retrieval import retrieve_relevant
    retrieved = retrieve_relevant(
        test_context,
        task_prompt,
        top_k=2,
        method="embedding",
        model_name=config.retrieval.model_name
    )
    
    console.print(f"✅ Retrieved {len(retrieved)} characters of context")
    console.print(f"\n📝 Retrieved context:\n{retrieved[:500]}...")
    
    # Now test with Hugging Face model
    generator = MultiLLMGenerator(config)
    hf_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    
    full_prompt = f"""You are a Python developer. Add a function to calculate the power of a number.

Context code:
{retrieved}

Task: Add a function `power(base, exponent)` that calculates base^exponent.
Return JSON:
```json
{{
    "files": {{
        "utils.py": "def power(base, exponent):\\n    return base ** exponent"
    }}
}}
```
"""
    
    try:
        console.print(f"\n🤖 Testing Hugging Face model with retrieval context...")
        response = await generator.generate_with_huggingface(hf_model, full_prompt)
        
        console.print(f"✅ Response received: {len(response)} characters")
        console.print(f"\n📝 Response:\n{response}")
        
        # Test parsing
        parsed = parse_llm_response(response, expected_language='python')
        if parsed:
            console.print(f"\n✅ Parsing successful: {len(parsed)} files extracted")
            for filename, content in parsed.items():
                console.print(f"  📄 {filename}:")
                console.print(f"  {content[:200]}...")
        else:
            console.print("⚠️ Parsing failed")
            
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        import traceback
        console.print(traceback.format_exc())


async def main():
    """Run all tests"""
    console.print(Panel.fit("🚀 Hugging Face Model Testing Suite", style="bold cyan"))
    
    # Test 1: Basic generation
    await test_hf_model_generation()
    
    console.print("\n" + "="*60 + "\n")
    
    # Test 2: Retrieval pipeline
    await test_retrieval_pipeline()
    
    console.print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
