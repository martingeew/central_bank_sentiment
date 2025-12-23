"""
Daily time series index builder.

Creates both forward-filled and sparse (non-filled) versions.
Merged from phase3_build_indices.py and phase3_build_indices_no_fill.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


def calculate_diffusion_index(values):
    """
    Calculate diffusion index for market impact variables.

    Formula: (% rise) + (0.5 * % neutral)

    Scale:
    - 100 = all 'rise'
    - 50 = neutral/mixed
    - 0 = all 'fall'

    Args:
        values: Series with 'rise', 'fall', or 'neutral' values

    Returns:
        Diffusion index (0-100)
    """
    if len(values) == 0:
        return np.nan

    counts = values.value_counts()
    total = len(values)

    pct_rise = counts.get('rise', 0) / total
    pct_neutral = counts.get('neutral', 0) / total

    diffusion = (pct_rise + 0.5 * pct_neutral) * 100

    return diffusion


def aggregate_daily_scores(df, institution):
    """
    Aggregate speeches to daily frequency for one institution.

    Args:
        df: DataFrame with all speeches
        institution: 'United States' or 'Euro area'

    Returns:
        DataFrame with daily indices (sparse - no forward fill)
    """
    inst_df = df[df['country'] == institution].copy()

    print(f"\n  Processing {institution}:")
    print(f"    Total speeches: {len(inst_df)}")
    print(f"    Date range: {inst_df['date'].min().date()} to {inst_df['date'].max().date()}")

    # Continuous metrics (calculate mean)
    continuous_metrics = [
        'hawkish_dovish_score',
        'uncertainty',
        'forward_guidance_strength',
        'topic_inflation',
        'topic_growth',
        'topic_financial_stability',
        'topic_labor_market',
        'topic_international'
    ]

    # Market impact variables (calculate diffusion index)
    market_metrics = [
        'market_impact_stocks',
        'market_impact_bonds',
        'market_impact_currency'
    ]

    # Group by date and aggregate
    daily_continuous = inst_df.groupby('date')[continuous_metrics].mean()

    # Calculate diffusion indices for market impact
    daily_market = inst_df.groupby('date')[market_metrics].agg(calculate_diffusion_index)

    # Rename market columns
    daily_market.columns = [col.replace('market_impact_', '') + '_diffusion_index'
                            for col in daily_market.columns]

    # Count speeches per day
    daily_counts = inst_df.groupby('date').size().rename('speech_count')

    # Combine all metrics
    daily_indices = pd.concat([daily_continuous, daily_market, daily_counts], axis=1)

    print(f"    Unique dates with speeches: {len(daily_indices)}")

    return daily_indices


def create_forward_filled(daily_indices):
    """
    Create continuous daily series with forward fill for missing dates.

    Args:
        daily_indices: DataFrame with indices for dates that have speeches

    Returns:
        DataFrame with all days filled (forward fill for gaps)
    """
    start_date = daily_indices.index.min()
    end_date = daily_indices.index.max()

    # Create full date range
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    print(f"    Creating continuous daily series:")
    print(f"      Start: {start_date.date()}")
    print(f"      End: {end_date.date()}")
    print(f"      Total days: {len(full_date_range)}")
    print(f"      Days with speeches: {len(daily_indices)}")
    print(f"      Days to fill: {len(full_date_range) - len(daily_indices)}")

    # Reindex to full date range
    full_indices = daily_indices.reindex(full_date_range)

    # Forward fill all columns except speech_count
    cols_to_fill = [col for col in full_indices.columns if col != 'speech_count']
    full_indices[cols_to_fill] = full_indices[cols_to_fill].ffill()

    # Fill speech_count with 0 for days with no speeches
    full_indices['speech_count'] = full_indices['speech_count'].fillna(0).astype(int)

    # Reset index to make date a column
    full_indices = full_indices.reset_index().rename(columns={'index': 'date'})

    return full_indices


class IndexBuilder:
    """
    Builds daily time series indices from sentiment results.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize index builder.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.indices_dir = Path(config['directories']['indices'])

    def build_indices(self, results_df: pd.DataFrame):
        """
        Build daily indices for Fed and ECB (both versions).

        Args:
            results_df: DataFrame with sentiment results
        """
        print("\n" + "=" * 70)
        print("BUILDING DAILY INDICES")
        print("=" * 70)

        # Ensure date column is datetime
        if 'date' not in results_df.columns:
            raise ValueError("results_df must have 'date' column")

        results_df['date'] = pd.to_datetime(results_df['date'])

        # Build for Fed
        print("\nFed indices:")
        fed_daily = aggregate_daily_scores(results_df, 'United States')

        # Save sparse version
        fed_no_fill = fed_daily.reset_index()
        fed_no_fill_file = self.indices_dir / "fed_daily_indices_no_fill.csv"
        fed_no_fill.to_csv(fed_no_fill_file, index=False)
        print(f"    Saved sparse version: {fed_no_fill_file.name}")

        # Save forward-filled version
        fed_filled = create_forward_filled(fed_daily)
        fed_filled_file = self.indices_dir / "fed_daily_indices.csv"
        fed_filled.to_csv(fed_filled_file, index=False)
        print(f"    Saved forward-filled version: {fed_filled_file.name}")

        # Build for ECB
        print("\nECB indices:")
        ecb_daily = aggregate_daily_scores(results_df, 'Euro area')

        # Save sparse version
        ecb_no_fill = ecb_daily.reset_index()
        ecb_no_fill_file = self.indices_dir / "ecb_daily_indices_no_fill.csv"
        ecb_no_fill.to_csv(ecb_no_fill_file, index=False)
        print(f"    Saved sparse version: {ecb_no_fill_file.name}")

        # Save forward-filled version
        ecb_filled = create_forward_filled(ecb_daily)
        ecb_filled_file = self.indices_dir / "ecb_daily_indices.csv"
        ecb_filled.to_csv(ecb_filled_file, index=False)
        print(f"    Saved forward-filled version: {ecb_filled_file.name}")

        print("\n" + "=" * 70)
        print("INDEX BUILDING COMPLETE")
        print("=" * 70)
        print(f"\nCreated 4 index files in: {self.indices_dir}")
        print("  - fed_daily_indices.csv (forward-filled)")
        print("  - fed_daily_indices_no_fill.csv (sparse)")
        print("  - ecb_daily_indices.csv (forward-filled)")
        print("  - ecb_daily_indices_no_fill.csv (sparse)")
