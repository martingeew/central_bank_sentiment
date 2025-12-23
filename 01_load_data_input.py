"""
Step 1: Load and Filter Speeches

Loads ECB-FED speeches from Hugging Face and creates a sample
based on date range specified in config.yaml.

Usage: python 01_load_data_input.py
Output: data/processed/sample_YYYY_YYYY.csv
"""

import utils
from data_loader import DataLoader


def main():
    """Main execution function."""
    utils.print_section_header("STEP 1: LOAD AND FILTER SPEECHES")

    # Load configuration
    print("\nLoading configuration...")
    config = utils.load_config()
    utils.ensure_directories(config)

    # Initialize data loader
    loader = DataLoader(config)

    # Load dataset from Hugging Face
    print("\nLoading ECB-FED speeches dataset...")
    speeches = loader.load_dataset()

    # Print summary
    loader.print_summary(speeches)

    # Filter by date range from config
    print("\nFiltering by date range...")
    date_range = config['date_range']
    print(f"  Start: {date_range['start']}")
    print(f"  End: {date_range['end']}")

    filtered = loader.filter_by_date_range(speeches)

    # Add speech_id column
    filtered['speech_id'] = [f"speech_{i}" for i in range(len(filtered))]

    # Save to processed folder
    start_year = date_range['start'][:4]
    end_year = date_range['end'][:4]
    output_file = f"{config['directories']['processed_data']}/sample_{start_year}_{end_year}.csv"

    print(f"\nSaving filtered speeches...")
    filtered.to_csv(output_file, index=False)

    print(f"\nSaved to: {output_file}")
    print(f"Total speeches: {len(filtered)}")

    if 'country' in filtered.columns:
        print("\nBreakdown by institution:")
        print(f"  Fed: {len(filtered[filtered['country']=='United States'])}")
        print(f"  ECB: {len(filtered[filtered['country']=='Euro area'])}")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE")
    print("=" * 70)
    print("\nNext step: python 02_make_indices.py")


if __name__ == "__main__":
    main()
