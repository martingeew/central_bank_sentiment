"""
Utility functions for Central Bank Sentiment Analysis.

Consolidated from src/utils.py and src/config.py.
"""

import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Dictionary with configuration settings
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required API keys
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError(
            "OPENAI_API_KEY not found. Please create .env file with your API key."
        )

    if not os.getenv('HF_TOKEN'):
        raise ValueError(
            "HF_TOKEN not found. Please add your Hugging Face token to .env file."
        )

    # Add API keys to config
    config['api_keys'] = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'huggingface': os.getenv('HF_TOKEN')
    }

    return config


def get_sentiment_prompt(speech_text: str, speaker: str, institution: str, date: str) -> str:
    """
    Create the sentiment analysis prompt for GPT-4.

    Args:
        speech_text: The speech content to analyze
        speaker: Name of the speaker
        institution: United States or Euro area
        date: Date of the speech

    Returns:
        Complete prompt string
    """
    prompt = f"""You are an expert in central bank communication analysis. Your task is to analyze speeches from Federal Reserve and ECB officials and extract monetary policy sentiment indicators used by financial market participants and economists.

**Speech Information:**
- Speaker: {speaker}
- Institution: {institution}
- Date: {date}

**Speech Text:**
{speech_text}

**Your Task:**
Analyze this speech and extract the following information. Be objective and base your analysis on the actual content.

**1. Hawkish/Dovish Score (-100 to +100):**
Rate the overall monetary policy stance:
- -100 = Extremely dovish (strongly favoring lower rates, more accommodation)
- 0 = Neutral (balanced, no clear directional bias)
- +100 = Extremely hawkish (strongly favoring higher rates, less accommodation)

Consider:
- Language about inflation (concerns = hawkish, moderating = dovish)
- Interest rate intentions (raising = hawkish, cutting = dovish)
- Economic risks (upside inflation risks = hawkish, downside growth risks = dovish)
- Urgency of action (immediate = more extreme, gradual = more moderate)

**2. Topic Emphasis (0-100 for each):**
How much emphasis does the speaker place on each topic?
- Inflation concerns: 0 (not mentioned) to 100 (major focus)
- Economic growth outlook: 0 to 100
- Financial stability risks: 0 to 100
- Labor market conditions: 0 to 100
- International/trade issues: 0 to 100

**3. Uncertainty Level (0-100):**
How certain is the speaker about the economic outlook and policy path?
- 0 = Very confident (clear statements, definitive language)
- 100 = Highly uncertain (many conditional statements, "depends on", "might", "could")

**4. Forward Guidance Strength (0-100):**
How explicit is guidance about future policy actions?
- 0 = No forward guidance (only describes current conditions)
- 50 = Implicit guidance (hints at future direction)
- 100 = Explicit commitment with timeline (e.g., "rates will remain elevated through 2024")

**5. Key Sentences:**
Extract 2-3 direct quotes that best support your hawkish/dovish score.

**6. Market Impact Prediction:**
Based on this speech, predict the likely immediate market reaction:
- Stock market: EXACTLY "rise" or "fall" or "neutral" (no other words allowed)
- Bond yields: EXACTLY "rise" or "fall" or "neutral" (no other words allowed)
- Currency (USD for Fed, EUR for ECB): EXACTLY "rise" or "fall" or "neutral"

CRITICAL: For currency, you MUST use ONLY these three exact words: "rise", "fall", or "neutral"
DO NOT use words like "strengthen", "weaken", "appreciate", "depreciate", "up", "down", etc.
ONLY use: "rise" (for strengthening), "fall" (for weakening), or "neutral"

- Brief reasoning: One sentence explaining your prediction

**Output Format:**
Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):

{{
  "hawkish_dovish_score": <number from -100 to 100>,
  "topics": {{
    "inflation": <number 0-100>,
    "growth": <number 0-100>,
    "financial_stability": <number 0-100>,
    "labor_market": <number 0-100>,
    "international": <number 0-100>
  }},
  "uncertainty": <number 0-100>,
  "forward_guidance_strength": <number 0-100>,
  "key_sentences": ["sentence 1", "sentence 2", "sentence 3"],
  "market_impact": {{
    "stocks": "rise",
    "bonds": "fall",
    "currency": "rise"
  }},
  "reasoning": "brief explanation",
  "summary": "One paragraph summarizing the main policy message and stance"
}}

IMPORTANT: The market_impact values MUST be exactly "rise", "fall", or "neutral" - no other words!"""

    return prompt


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.

    Rough approximation: 1 token = 4 characters

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def calculate_cost(input_tokens: int, output_tokens: int, config: Dict[str, Any]) -> float:
    """
    Calculate API cost based on token usage and config pricing.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        config: Configuration dictionary with pricing info

    Returns:
        Total cost in USD
    """
    pricing = config['pricing']

    # Convert per-1M pricing to per-token
    input_cost = (input_tokens / 1_000_000) * pricing['input_per_1m_tokens']
    output_cost = (output_tokens / 1_000_000) * pricing['output_per_1m_tokens']

    # Apply batch discount
    total_cost = (input_cost + output_cost) * pricing['batch_discount']

    return total_cost


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost * 100:.4f}c"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def save_json(data: Dict[Any, Any], file_path: Path):
    """Save dictionary to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: Path) -> Dict[Any, Any]:
    """Load dictionary from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_section_header(title: str, width: int = 70):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def ensure_directories(config: Dict[str, Any]):
    """
    Create all required directories from config.

    Args:
        config: Configuration dictionary
    """
    dirs = config['directories']
    for key, path in dirs.items():
        Path(path).mkdir(parents=True, exist_ok=True)
