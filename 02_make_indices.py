"""
Step 2: Process Batches and Build Indices

1. Creates batch files with chunking (stays under 90k token limit)
2. Submits to OpenAI Batch API
3. Monitors progress and downloads results
4. Validates outputs
5. Builds daily time series indices (forward-filled & sparse)

Usage: python 02_make_indices.py
Input: data/processed/sample_*.csv
Output:
  - outputs/batch_files/*.jsonl
  - outputs/batch_results/chunk*_results.jsonl
  - outputs/indices/fed_daily_indices.csv
  - outputs/indices/ecb_daily_indices.csv
  - outputs/reports/validation_report.json
"""

import pandas as pd
import utils
from batch_processor import BatchProcessor
from index_builder import IndexBuilder
from output_validator import OutputValidator


def main():
    """Main execution function."""
    utils.print_section_header("STEP 2: BATCH PROCESSING & INDEX BUILDING")

    # Load configuration
    print("\nLoading configuration...")
    config = utils.load_config()
    utils.ensure_directories(config)

    # Find sampled speeches file
    date_range = config['date_range']
    start_year = date_range['start'][:4]
    end_year = date_range['end'][:4]
    speeches_file = f"{config['directories']['processed_data']}/sample_{start_year}_{end_year}.csv"

    print(f"\nLoading speeches from: {speeches_file}")
    speeches_df = pd.read_csv(speeches_file, parse_dates=['date'])
    print(f"Total speeches: {len(speeches_df)}")

    # Initialize batch processor
    processor = BatchProcessor(config)

    # Step 2a: Create chunked batch files
    utils.print_section_header("CREATE BATCH FILES")
    batch_files = processor.create_chunked_batch_files(speeches_df)

    # Estimate cost
    total_speeches = len(speeches_df)
    avg_speech_length = speeches_df['content'].str.len().mean()
    estimated_input_tokens = total_speeches * (utils.estimate_tokens(str(avg_speech_length)) + 1000)  # +1000 for prompt
    estimated_output_tokens = total_speeches * 500  # ~500 tokens per response

    cost = utils.calculate_cost(estimated_input_tokens, estimated_output_tokens, config)
    print(f"\nEstimated cost:")
    print(f"  Input tokens: ~{estimated_input_tokens:,}")
    print(f"  Output tokens: ~{estimated_output_tokens:,}")
    print(f"  Total cost: {utils.format_cost(cost)}")

    # Ask user confirmation
    proceed = input("\nProceed with batch submission? (y/n): ")
    if proceed.lower() != 'y':
        print("Aborted")
        return

    # Step 2b: Submit batches
    utils.print_section_header("SUBMIT BATCHES")
    submit_only = input("\nSubmit without waiting for completion? (y/n): ").lower() == 'y'

    batch_info = processor.process_all_chunks(batch_files, submit_only=submit_only)

    # Save batch info
    utils.save_json(batch_info, f"{config['directories']['reports']}/batch_info.json")

    if submit_only:
        print("\nBatches submitted. Run this script again later to download results.")
        return

    # Step 2c: Combine chunk results
    utils.print_section_header("COMBINE RESULTS")
    combined_results = processor.combine_chunk_results(speeches_df)

    # Save combined results
    results_file = f"{config['directories']['processed_data']}/sentiment_results_{start_year}_{end_year}.csv"
    combined_results.to_csv(results_file, index=False)
    print(f"\nSaved combined results: {results_file}")

    # Step 2d: Validate outputs
    utils.print_section_header("VALIDATE OUTPUTS")
    validator = OutputValidator()
    validation_results = validator.validate_batch_results(combined_results)
    validator.print_validation_report(validation_results)

    # Save validation report
    utils.save_json(validation_results, f"{config['directories']['reports']}/validation_report.json")

    # Step 2e: Build daily indices
    utils.print_section_header("BUILD DAILY INDICES")
    builder = IndexBuilder(config)
    builder.build_indices(combined_results)

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {config['directories']['indices']}/")
    print("\nNext step: python 03_visualize_indices.py")


if __name__ == "__main__":
    main()
