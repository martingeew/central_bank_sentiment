# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **central bank sentiment analysis pipeline** that extracts 5 sentiment dimensions from Federal Reserve and ECB speeches using OpenAI's Batch API. It processes speeches through a multi-stage pipeline: data loading → LLM batch processing → index building → visualization.

**Key features:**
- Analyzes hawkish/dovish stance, topic emphasis, uncertainty, forward guidance, and market impact predictions
- Uses OpenAI Batch API with automatic chunking to handle 90k token limit
- Creates daily time series indices (both forward-filled and sparse versions)
- Generates 18 visualization types (bar charts, area plots, calendar heatmaps)

## Development Commands

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# Step 1: Load and filter speeches by date range
python 01_load_data_input.py

# Step 2: Process batches and build indices (requires OpenAI API key)
python 02_make_indices.py

# Step 3: Create visualizations
python 03_visualize_indices.py
```

### Configuration
All settings are in `config.yaml`:
- **Date range**: `date_range.start` and `date_range.end`
- **Model**: `model.name` (e.g., "gpt-4o-2024-08-06", "gpt-4o-mini")
- **Token limits**: `chunking.max_tokens_per_chunk` (default: 75000)
- **Visualization toggles**: `charts.create_bars/areas/calendars`

API keys must be in `.env`:
```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

## Architecture

### Pipeline Flow

```
Raw Data (HuggingFace)
  → 01_load_data_input.py (DataLoader)
  → data/processed/sample_YYYY_YYYY.csv
  → 02_make_indices.py (BatchProcessor → IndexBuilder)
  → outputs/indices/*.csv
  → 03_visualize_indices.py (Visualizer)
  → outputs/charts/*.png
```

### Core Modules

**data_loader.py** - Hugging Face dataset integration
- Downloads/caches `istat-ai/ECB-FED-speeches` dataset
- Filters by date range from config
- Stores local cache in `data/raw/ecb_fed_speeches.parquet`

**batch_processor.py** - OpenAI Batch API handler
- **Chunking logic**: Splits speeches into chunks based on token estimates (not fixed count)
- Iterates through speeches, estimates tokens per request, groups into chunks ≤ max_tokens_per_chunk
- Creates JSONL files in `outputs/batch_files/chunk{N:02d}_input.jsonl`
- Submits batches, monitors status, downloads results
- Key method: `create_chunked_batch_files()` returns `(batch_files, estimated_input_tokens)`

**index_builder.py** - Daily time series aggregation
- Aggregates speeches to daily frequency by institution (Fed/ECB)
- Continuous metrics (hawkish/dovish, uncertainty, topics): uses mean aggregation
- Market impact (stocks/bonds/currency): uses diffusion index formula: `(% rise) + (0.5 * % neutral) * 100`
- Creates **two versions**:
  - Forward-filled: gaps filled with last value (for trend analysis)
  - Sparse: only dates with actual speeches (for event analysis)

**output_validator.py** - LLM output validation
- Validates all required fields exist and are in correct ranges
- Scores must be numeric: hawkish_dovish_score [-100, 100], topics/uncertainty/forward_guidance [0, 100]
- Market impact values MUST be exactly "rise", "fall", or "neutral"
- Generates validation report showing % of valid outputs and error types

**visualizer.py** - Chart generation
- Bar charts: uses sparse data (speech dates only)
- Area plots: uses forward-filled data (continuous time series)
- Calendar heatmaps: uses sparse data with dayplot library
- Special handling: hawkish/dovish uses diverging colormap, diffusion indices centered at 50

**utils.py** - Shared utilities
- `get_sentiment_prompt()`: Creates LLM prompt for sentiment extraction (critical for accuracy)
- `estimate_tokens()`: Rough approximation (1 token ≈ 4 chars)
- `calculate_cost()`: Estimates API cost with batch discount
- `load_config()`: Loads YAML config and validates API keys from .env

### Data Flow Details

**Speech Columns** (from Hugging Face → after processing):
- Raw: `date, author, country, text`
- After 01: adds `speech_id` (e.g., "speech_0")
- After 02: adds sentiment fields (hawkish_dovish_score, topic_*, uncertainty, forward_guidance_strength, market_impact_*)

**Index Files** (outputs/indices/):
- `fed_daily_indices.csv` / `ecb_daily_indices.csv`: Forward-filled daily series
- `fed_daily_indices_no_fill.csv` / `ecb_daily_indices_no_fill.csv`: Sparse (speech dates only)
- Columns: date, hawkish_dovish_score, uncertainty, forward_guidance_strength, topic_inflation/growth/financial_stability/labor_market/international, stocks/bonds/currency_diffusion_index, speech_count

**Batch Processing States**:
1. Create chunks → JSONL files in batch_files/
2. Upload to OpenAI → get batch IDs
3. Poll status every 60s (validating → in_progress → completed/failed)
4. Download results → chunk*_results.jsonl in batch_results/
5. Parse and combine → sentiment_results_*.csv

### Token Chunking Strategy

**Why chunking is needed**: OpenAI Batch API has 90,000 enqueued token limit per batch.

**How it works**:
1. For each speech, create batch request and estimate tokens
2. Group requests into chunks where sum(tokens) ≤ max_tokens_per_chunk
3. If adding a request exceeds limit, start new chunk
4. Default max_tokens_per_chunk: 75,000 (safety buffer under 90k limit)

**Critical**: Token estimation is approximate (text_length // 4). If batches still fail with token errors, reduce `chunking.max_tokens_per_chunk` in config.yaml.

## Common Modifications

### Changing the LLM Prompt
Edit `utils.py:get_sentiment_prompt()`. The prompt includes:
- System instructions on what to extract
- Score definitions and ranges
- Output format (JSON structure)
- Market impact constraints (MUST use "rise"/"fall"/"neutral" only)

If you modify the prompt:
1. Update `output_validator.py` validation rules to match
2. Update `batch_processor.py:parse_results()` if JSON structure changes
3. Re-run full pipeline (01 → 02 → 03)

### Adding New Metrics
To add a new sentiment dimension:
1. Update prompt in `utils.py:get_sentiment_prompt()`
2. Add parsing in `batch_processor.py:parse_results()`
3. Add validation in `output_validator.py:validate_single_output()`
4. Add aggregation in `index_builder.py:aggregate_daily_scores()`
5. Add visualization in `visualizer.py` (choose appropriate colormap in `get_colormap_settings()`)

### Adjusting Date Range
Edit `config.yaml:date_range` and re-run all 3 scripts. Cost scales linearly (~$7 per 2 years for gpt-4o).

### Switching Models
Edit `config.yaml:model.name` and update `pricing` section to match OpenAI pricing page.

## File Organization

```
central_bank_sentiment/
├── config.yaml              # All configuration
├── .env                     # API keys (not in git)
├── 01_load_data_input.py   # Step 1: Load speeches
├── 02_make_indices.py      # Step 2: Batch processing & indexing
├── 03_visualize_indices.py # Step 3: Create charts
├── utils.py                 # Shared utilities + LLM prompt
├── data_loader.py           # Hugging Face integration
├── batch_processor.py       # OpenAI Batch API logic
├── index_builder.py         # Daily aggregation + diffusion indices
├── output_validator.py      # LLM output validation
├── visualizer.py            # Chart generation
├── data/
│   ├── raw/                 # Cached HF dataset (auto-downloaded)
│   └── processed/           # Filtered speeches + sentiment results
└── outputs/
    ├── batch_files/         # JSONL batch inputs (chunked)
    ├── batch_results/       # JSONL batch outputs from OpenAI
    ├── indices/             # Daily time series CSVs (4 files)
    ├── charts/              # Visualizations (18 PNG files)
    └── reports/             # Validation + batch metadata
```

## Important Notes

**Token Limit Errors**: If you see "Enqueued token limit reached", reduce `chunking.max_tokens_per_chunk` in config.yaml. The chunking algorithm groups speeches dynamically based on token estimates.

**Validation Rate**: Aim for >95% validation rate. If lower, check `outputs/reports/validation_report.json` for error types. Common issues:
- Market impact using wrong words (must be "rise"/"fall"/"neutral")
- Scores out of range (LLM hallucination)
- Missing required fields (JSON parsing error)

**Forward-Fill vs Sparse**: Use forward-filled indices for trend analysis (continuous daily series). Use sparse indices for event studies (only speech dates). Bar charts and calendar heatmaps always use sparse data.

**Diffusion Index Formula**: For market impact, we calculate `(pct_rise + 0.5 * pct_neutral) * 100`. This gives 100=all rise, 50=neutral/mixed, 0=all fall.

**Batch API Timing**: OpenAI processes batches within 24 hours, but typically completes in 30min-4 hours. Use `submit_only=True` option in step 2 to submit and exit, then re-run later to download results.
