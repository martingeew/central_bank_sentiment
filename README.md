# Central Bank Sentiment Analysis

Analyze Federal Reserve and ECB speeches using OpenAI's Batch API to extract sentiment, policy stance, and market impact predictions.

## What This Does

- **Extracts 5 sentiment dimensions** from central bank speeches: hawkish/dovish score, topic emphasis, uncertainty, forward guidance, and market impact
- **Uses OpenAI Batch API** for cost-effective processing (50% discount vs real-time API)
- **Automatic chunking** to handle OpenAI's 90,000 token limit per batch
- **Creates daily indices** showing how central bank sentiment evolves over time
- **Generates 18 visualizations** including bar charts, area plots, and calendar heatmaps

## Prerequisites

**Required:**
- Python 3.9 or higher
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Hugging Face token ([get one here](https://huggingface.co/settings/tokens) - select "Read" access)

**Recommended:**
- 2-4 GB free disk space (for dataset cache)
- Stable internet connection

## Quick Start

### 1. Download and Setup

```bash
# Clone or download this repository
cd central_bank_sentiment

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# On Windows: notepad .env
# On macOS/Linux: nano .env
```

Add your keys to the `.env` file:
```
OPENAI_API_KEY=sk-your-actual-openai-key-here
HF_TOKEN=hf_your-actual-huggingface-token-here
```

### 3. Customize Settings (Optional)

Edit `config.yaml` to change:
- **Date range**: Change `date_range.start` and `date_range.end`
- **Model**: Change `model.name` (e.g., to `gpt-4o-mini` for lower cost)
- **Cost estimates**: Update `pricing` section based on current OpenAI pricing

### 4. Run the Analysis

```bash
# Step 1: Load and filter speeches (takes 2-5 minutes)
python 01_load_data_input.py

# Step 2: Process batches and build indices (takes 30 min - 4 hours)
python 02_make_indices.py           # Submit all chunks and wait for completion
python 02_make_indices.py --chunk 5 # Submit only chunk 5 (for testing/retry)
python 02_make_indices.py --resume  # Resume from existing batches (check status, download completed, resubmit failed)

# Step 3: Create visualizations (takes 1-2 minutes)
python 03_visualize_indices.py
```

**Step 2 options explained:**
- **Default** (no arguments): Submits all chunks. You'll be asked if you want to wait for completion or submit and check later.
- **`--chunk N`**: Submit only a specific chunk number. Useful for resubmitting individual failed chunks.
- **`--resume`**: Check status of existing batches, download completed results, and optionally resubmit any failed batches.

### 5. View Results

Your outputs will be in these folders:
- `outputs/indices/` - Daily time series CSV files
- `outputs/charts/` - PNG visualization files
- `outputs/reports/` - Validation reports

## Configuration Guide

### Changing the Date Range

Edit `config.yaml`:

```yaml
date_range:
  start: "2020-01-01"  # Change to your desired start date
  end: "2025-12-31"    # Change to your desired end date
```

Then re-run all 3 scripts.

### Switching Models

To use a different GPT model, edit `config.yaml`:

```yaml
model:
  name: "gpt-4o-mini"  # Cheaper alternative
  # name: "gpt-4-turbo"  # More expensive, may be more accurate
```

**Important**: Also update the pricing section to match the model's cost:

```yaml
pricing:
  input_per_1m_tokens: 0.15   # Update based on OpenAI pricing page
  output_per_1m_tokens: 0.60
  batch_discount: 0.50        # Batch API is 50% off
```

Check current pricing: https://openai.com/api/pricing/

### Understanding Chunking

The script automatically splits your data into chunks to stay under OpenAI's 90,000 enqueued token limit.

**Why chunking is needed:**
```
Error you'll get without chunking:
"Enqueued token limit reached for gpt-4o-2024-08-06. Limit: 90,000 enqueued tokens"
```

**How it works:**
- 311 speeches (2022-2023 sample) = ~5 million tokens total
- Automatically split into 17 chunks (~18-20 speeches each)
- Each chunk ~5,000 tokens (well under 90,000 limit)
- Chunks process in parallel on OpenAI's servers
- Results automatically combined

**Adjusting chunk size** (only if needed):

```yaml
chunking:
  max_tokens_per_chunk: 75000  # Stay under OpenAI's 90k limit
```

The script uses **token-based chunking** (not count-based) to reliably stay under the 90,000 token limit. Each chunk's size is estimated based on actual token count rather than number of speeches.

## Understanding the Outputs

### Daily Indices Files

**Forward-filled versions** (for trend analysis):
- `fed_daily_indices.csv` - Continuous daily series
- `ecb_daily_indices.csv` - Continuous daily series
- Last value carried forward to fill gaps (no speeches on weekends/holidays)

**Sparse versions** (for accurate speech-date representation):
- `fed_daily_indices_no_fill.csv` - Only dates with actual speeches
- `ecb_daily_indices_no_fill.csv` - Only dates with actual speeches

### CSV Columns

**Policy metrics:**
- `hawkish_dovish_score` (-100 to +100): Tightening stance (positive = hawkish)
- `uncertainty` (0-100): How uncertain about economic outlook
- `forward_guidance_strength` (0-100): How explicit about future policy

**Topic emphasis** (0-100 each):
- `topic_inflation` - Focus on inflation concerns
- `topic_growth` - Focus on economic growth
- `topic_financial_stability` - Focus on financial risks
- `topic_labor_market` - Focus on employment
- `topic_international` - Focus on international issues

**Market impact diffusion indices** (0-100):
- `stocks_diffusion_index` - Stock market prediction (100=all rise, 0=all fall, 50=mixed)
- `bonds_diffusion_index` - Bond yields prediction
- `currency_diffusion_index` - Currency prediction

### Visualization Files

**Bar charts** (`*_bars.png`):
- Shows individual speeches as bars
- Good for seeing specific events

**Area plots** (`*_area.png`):
- Shows trends over time
- Uses continuous data (forward-filled)

**Calendar heatmaps** (`*_calendar.png`):
- Shows sentiment in calendar format
- Grey cells = no speech that day

## Cost Estimation

For the 2022-2023 sample (311 speeches):

```
Estimated cost with gpt-4o-2024-08-06:
- Input: ~5M tokens × $2.50/1M = $12.50
- Output: ~150K tokens × $10.00/1M = $1.50
- Subtotal: $14.00
- Batch discount (50%): $7.00
- Total: ~$7.00
```

For different date ranges, costs scale roughly linearly:
- 1 year of speeches: ~$3.50
- 5 years of speeches: ~$17.50
- 10 years of speeches: ~$35.00

**Cost-saving tips:**
1. Use `gpt-4o-mini` (about 90% cheaper)
2. Start with shorter date ranges for testing
3. Always use Batch API (automatic 50% discount)

## Common Issues

### Issue: `OPENAI_API_KEY not found`

**Solution**: Make sure you created `.env` file (not `.env.example`) and added your actual API key.

```bash
# Check if file exists
ls .env  # Should show .env, not .env.example

# Check file contents (your key should be there)
cat .env  # On Windows: type .env
```

### Issue: `Enqueued token limit reached`

**Solution**: This is why we use chunking! If you still get this error, you're trying to process too much at once.

- Check `config.yaml` → `chunking.max_speeches_per_chunk`
- Make sure it's 20 or less
- If running multiple batch jobs, wait for previous ones to complete

### Issue: `HF_TOKEN not found` or `401 Unauthorized`

**Solution**:
1. Go to https://huggingface.co/settings/tokens
2. Create a token with "Read" access
3. Add it to your `.env` file
4. Make sure there are no extra spaces or quotes

### Issue: Download is very slow

**Solution**: First download takes time (2-3 GB dataset). Subsequent runs use cached version stored in `data/raw/`.

### Issue: Batch processing takes longer than expected

**Solution**: OpenAI processes batches within 24 hours, but often completes in 30 minutes to 4 hours. If you don't want to wait:

```bash
# In step 2, when asked:
"Submit without waiting for completion? (y/n):"
# Answer: y

# Then later, run step 2 again to download results
python 02_make_indices.py
```

### Issue: `insufficient_quota` error

**Error message:**
```json
"status_code": 429,
"request_id": "14991783263bcb60ffd802bfeea34bf2",
"body": {
  "error": {
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.",
    "type": "insufficient_quota",
    "param": null,
    "code": "insufficient_quota"
  }
}
```

**Solution**: Your OpenAI account credits should be topped up appropriately according to the cost estimates shown when you run step 2.

1. Check your current balance: https://platform.openai.com/account/billing
2. Add credits based on the estimated cost displayed
3. If the batch was interrupted, use `python 02_make_indices.py --resume` to continue

### Issue: Validation rate below 95%

**Solution**: Check `outputs/reports/validation_report.json` to see what errors occurred. Common issues:
- Model returned incorrect format (try different model)
- Network errors during batch processing (re-run step 2)
- Invalid market impact values (check if model follows "rise/fall/neutral" constraint)

### Issue: Charts not created

**Solution**:
1. Check that step 2 completed successfully
2. Verify `outputs/indices/*.csv` files exist
3. For calendar heatmaps, ensure `dayplot` is installed: `pip install dayplot`

## Project Structure

```
central_bank_sentiment/
├── config.yaml              # All settings
├── .env                     # API keys (you create this)
├── requirements.txt         # Python dependencies
│
├── 01_load_data_input.py   # Step 1: Load speeches
├── 02_make_indices.py      # Step 2: Process & build indices
├── 03_visualize_indices.py # Step 3: Create charts
│
├── utils.py                # Helper functions
├── data_loader.py          # Hugging Face dataset loader
├── batch_processor.py      # OpenAI Batch API handler
├── index_builder.py        # Daily indices builder
├── visualizer.py           # Chart creator
├── output_validator.py     # Output validation
│
├── data/
│   ├── raw/                # Downloaded dataset (auto-created)
│   └── processed/          # Filtered speeches
│
└── outputs/
    ├── batch_files/        # Generated JSONL files
    ├── batch_results/      # Raw API responses
    ├── indices/            # Daily time series CSV
    ├── charts/             # Visualizations
    └── reports/            # Validation reports
```

## Advanced Customization

### Modifying the Sentiment Prompt

To change what the LLM extracts, edit `utils.py` → `get_sentiment_prompt()` function.

### Adding New Metrics

1. Update prompt in `utils.py`
2. Update parsing in `batch_processor.py` → `parse_results()`
3. Update aggregation in `index_builder.py`
4. Add visualization in `visualizer.py`

### Processing Full Dataset

To analyze all speeches (1996-2025):

```yaml
# config.yaml
date_range:
  start: "1996-01-01"
  end: "2025-12-31"
```

**Warning**: This will cost ~$100-150 and take several hours.

## License

See LICENSE file for details.

## Acknowledgments

- **Dataset**: ECB-FED speeches from [istat-ai/ECB-FED-speeches](https://huggingface.co/datasets/istat-ai/ECB-FED-speeches)
- **API**: OpenAI's GPT-4o and Batch API
- **Visualization**: dayplot library for calendar heatmaps

## Questions?

If you encounter issues not covered here, check:
1. OpenAI API status: https://status.openai.com/
2. Hugging Face status: https://status.huggingface.co/
3. Your OpenAI usage limits: https://platform.openai.com/usage

---

**Last Updated**: December 2025
