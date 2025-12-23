"""
Data loader for ECB-FED speeches dataset from Hugging Face.

Simplified version for blog readers.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
from huggingface_hub import login
from typing import Dict, Any


class DataLoader:
    """
    Handles loading and filtering ECB-FED speeches from Hugging Face.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.raw_data_dir = Path(config['directories']['raw_data'])
        self.hf_token = config['api_keys']['huggingface']

        # Authenticate with Hugging Face
        if self.hf_token:
            try:
                login(token=self.hf_token, add_to_git_credential=False)
                print("Authenticated with Hugging Face")
            except Exception as e:
                print(f"Warning: Hugging Face authentication issue: {e}")

    def load_dataset(self, force_download: bool = False) -> pd.DataFrame:
        """
        Load ECB-FED speeches dataset.

        Args:
            force_download: If True, download even if local cache exists

        Returns:
            DataFrame with all speeches
        """
        local_file = self.raw_data_dir / "ecb_fed_speeches.parquet"

        # Load from cache if exists
        if local_file.exists() and not force_download:
            print(f"Loading dataset from: {local_file}")
            df = pd.read_parquet(local_file)
            print(f"Loaded {len(df)} speeches from local storage")
            return df

        # Download from Hugging Face
        print("Downloading dataset from Hugging Face: istat-ai/ECB-FED-speeches")
        print("This may take a few minutes on first run...")

        try:
            dataset = load_dataset(
                "istat-ai/ECB-FED-speeches",
                token=self.hf_token
            )

            # Convert to pandas
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()

            print(f"Downloaded {len(df)} speeches")

            # Save to local cache
            print(f"Saving to: {local_file}")
            df.to_parquet(local_file, index=False)
            print("Saved successfully")

            return df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def filter_by_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter speeches by date range from config.

        Args:
            df: Full speeches DataFrame

        Returns:
            Filtered DataFrame
        """
        date_range = self.config['date_range']
        start_date = pd.to_datetime(date_range['start'])
        end_date = pd.to_datetime(date_range['end'])

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError("No date column found in dataset")

        # Filter
        filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

        print(f"\nFiltered speeches from {start_date.date()} to {end_date.date()}")
        print(f"Total speeches: {len(filtered)}")

        if 'country' in filtered.columns:
            print(f"  Fed: {len(filtered[filtered['country']=='United States'])}")
            print(f"  ECB: {len(filtered[filtered['country']=='Euro area'])}")

        return filtered

    def print_summary(self, df: pd.DataFrame):
        """Print dataset summary."""
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)

        print(f"\nTotal speeches: {len(df)}")

        if 'country' in df.columns:
            print("\nBy institution:")
            for inst, count in df['country'].value_counts().items():
                print(f"  {inst}: {count}")

        if 'date' in df.columns:
            print(f"\nDate range:")
            print(f"  Earliest: {df['date'].min().date()}")
            print(f"  Latest: {df['date'].max().date()}")

        if 'year' in df.columns:
            print("\nBy year:")
            for year in sorted(df['year'].unique(), reverse=True)[:5]:
                count = len(df[df['year'] == year])
                print(f"  {year}: {count}")

        print("=" * 70)
