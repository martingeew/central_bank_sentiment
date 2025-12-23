"""
Visualization module for central bank sentiment indices.

Merges all three visualization types:
- Bar charts (sparse data)
- Area plots (forward-filled data)
- Calendar heatmaps (sparse data)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import seaborn.objects as so
import dayplot as dp
from pathlib import Path
from typing import Dict, Any


def get_colormap_settings(var_name):
    """
    Get colormap and vcenter settings for a variable.

    Args:
        var_name: Variable name from dataframe

    Returns:
        Tuple of (cmap, vcenter, vmin, vmax)
    """
    if var_name == "hawkish_dovish_score":
        return ("coolwarm", 0, -100, 100)
    elif var_name in [
        "stocks_diffusion_index",
        "bonds_diffusion_index",
        "currency_diffusion_index",
    ]:
        return ("coolwarm", 50, 0, 100)
    elif var_name.startswith("topic_"):
        return ("YlOrRd", None, -1, 100)
    elif var_name in ["uncertainty", "forward_guidance_strength"]:
        return ("Greens", None, -1, 100)
    elif var_name == "speech_count":
        base_cmap = plt.cm.get_cmap("Greens")
        colors = base_cmap(np.linspace(0.3, 1.0, 256))
        dark_greens = mcolors.LinearSegmentedColormap.from_list("DarkGreens", colors)
        return (dark_greens, None, 1, 6)
    else:
        return ("Greens", None, 0, 100)


def format_metric_title(var_name):
    """
    Format variable name into a readable title.

    Args:
        var_name: Variable name from dataframe

    Returns:
        Formatted title string
    """
    title = var_name.replace("_", " ").title()

    replacements = {
        "Hawkish Dovish": "Hawkish/Dovish",
        "Diffusion Index": "Diffusion",
        "Topic ": "",
        "Score": "",
    }

    for old, new in replacements.items():
        title = title.replace(old, new)

    return title.strip()


class Visualizer:
    """
    Creates all visualizations for sentiment indices.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.charts_dir = Path(config['directories']['charts'])
        self.indices_dir = Path(config['directories']['indices'])

    def load_indices(self):
        """Load all index files."""
        indices = {
            'fed_filled': pd.read_csv(self.indices_dir / "fed_daily_indices.csv", parse_dates=['date']),
            'fed_sparse': pd.read_csv(self.indices_dir / "fed_daily_indices_no_fill.csv", parse_dates=['date']),
            'ecb_filled': pd.read_csv(self.indices_dir / "ecb_daily_indices.csv", parse_dates=['date']),
            'ecb_sparse': pd.read_csv(self.indices_dir / "ecb_daily_indices_no_fill.csv", parse_dates=['date'])
        }
        return indices

    def create_bar_charts(self):
        """Create bar chart visualizations (uses sparse data)."""
        if not self.config['charts']['create_bars']:
            return

        print("\nCreating bar charts...")
        indices = self.load_indices()

        for inst, inst_name in [('fed', 'Federal Reserve'), ('ecb', 'European Central Bank')]:
            df = indices[f'{inst}_sparse']

            # Policy metrics
            self._create_policy_bars(df, inst, inst_name)

            # Topic indices
            self._create_topic_bars(df, inst, inst_name)

            # Market impact
            self._create_market_bars(df, inst, inst_name)

        print("  Bar charts complete")

    def _create_policy_bars(self, df, inst, inst_name):
        """Create policy metrics bar chart."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{inst_name} - Policy Metrics', fontsize=16, fontweight='bold')

        # Hawkish/Dovish (diverging)
        ax = axes[0, 0]
        colors = ['#d62728' if x < 0 else '#1f77b4' for x in df['hawkish_dovish_score']]
        ax.bar(df['date'], df['hawkish_dovish_score'], color=colors, width=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Hawkish/Dovish Score')
        ax.set_ylim(-100, 100)

        # Uncertainty
        ax = axes[0, 1]
        ax.bar(df['date'], df['uncertainty'], color='#ff7f0e', width=1.5)
        ax.set_title('Uncertainty Level')
        ax.set_ylim(0, 100)

        # Forward Guidance
        ax = axes[1, 0]
        ax.bar(df['date'], df['forward_guidance_strength'], color='#2ca02c', width=1.5)
        ax.set_title('Forward Guidance Strength')
        ax.set_ylim(0, 100)

        # Speech count
        ax = axes[1, 1]
        ax.bar(df['date'], df['speech_count'], color='#9467bd', width=1.5)
        ax.set_title('Speeches per Day')

        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{inst}_policy_metrics_bars.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_topic_bars(self, df, inst, inst_name):
        """Create topic indices bar chart."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'{inst_name} - Topic Emphasis', fontsize=16, fontweight='bold')

        topics = [
            ('topic_inflation', 'Inflation'),
            ('topic_growth', 'Economic Growth'),
            ('topic_financial_stability', 'Financial Stability'),
            ('topic_labor_market', 'Labor Market'),
            ('topic_international', 'International Issues')
        ]

        palette = sns.color_palette()

        for idx, (col, title) in enumerate(topics):
            row = idx // 3
            col_idx = idx % 3
            ax = axes[row, col_idx]
            ax.bar(df['date'], df[col], color=palette[idx], width=1.5)
            ax.set_title(title)
            ax.set_ylim(0, 100)

        # Hide last subplot
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{inst}_topic_indices_bars.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_market_bars(self, df, inst, inst_name):
        """Create market impact diffusion index bars."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'{inst_name} - Market Impact Diffusion Indices', fontsize=16, fontweight='bold')

        markets = [
            ('stocks_diffusion_index', 'Stock Market'),
            ('bonds_diffusion_index', 'Bond Yields'),
            ('currency_diffusion_index', f"{'USD' if inst == 'fed' else 'EUR'}")
        ]

        for idx, (col, title) in enumerate(markets):
            ax = axes[idx]
            colors = ['#C41E28' if x < 50 else '#048060' for x in df[col]]
            ax.bar(df['date'], df[col], color=colors, width=1.5)
            ax.axhline(y=50, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_title(title)
            ax.set_ylim(0, 100)
            ax.grid(False)
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{inst}_market_impact_bars.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_area_plots(self):
        """Create area plot visualizations (uses forward-filled data)."""
        if not self.config['charts']['create_areas']:
            return

        print("\nCreating area plots...")
        indices = self.load_indices()

        for inst, inst_name in [('fed', 'Federal Reserve'), ('ecb', 'European Central Bank')]:
            df = indices[f'{inst}_filled']
            self._create_policy_areas(df, inst, inst_name)
            self._create_topic_areas(df, inst, inst_name)

        print("  Area plots complete")

    def _create_policy_areas(self, df, inst, inst_name):
        """Create policy metrics area plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{inst_name} - Policy Metrics (Continuous)', fontsize=16, fontweight='bold')

        # Hawkish/Dovish (special diverging colors)
        ax = axes[0, 0]
        palette = sns.color_palette()
        blue = palette[0]
        red = palette[3]
        ax.fill_between(df['date'], 0, df['hawkish_dovish_score'],
                        where=(df['hawkish_dovish_score'] >= 0), color=red, alpha=0.3)
        ax.fill_between(df['date'], 0, df['hawkish_dovish_score'],
                        where=(df['hawkish_dovish_score'] < 0), color=blue, alpha=0.3)
        ax.plot(df['date'], df['hawkish_dovish_score'], color='#888888', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Hawkish/Dovish Score')
        ax.set_ylim(-100, 100)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Other metrics
        metrics = [
            ('uncertainty', 'Uncertainty Level', '#ff7f0e', axes[0, 1]),
            ('forward_guidance_strength', 'Forward Guidance', '#2ca02c', axes[1, 0])
        ]

        for col, title, color, ax in metrics:
            ax.fill_between(df['date'], 0, df[col], color=color, alpha=0.3)
            ax.plot(df['date'], df[col], color=color, linewidth=2)
            ax.set_title(title)
            ax.set_ylim(0, 100)
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{inst}_policy_metrics_area.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_topic_areas(self, df, inst, inst_name):
        """Create topic emphasis area plots."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'{inst_name} - Topic Emphasis (Continuous)', fontsize=16, fontweight='bold')

        topics = [
            ('topic_inflation', 'Inflation'),
            ('topic_growth', 'Economic Growth'),
            ('topic_financial_stability', 'Financial Stability'),
            ('topic_labor_market', 'Labor Market'),
            ('topic_international', 'International Issues')
        ]

        palette = sns.color_palette()

        for idx, (col, title) in enumerate(topics):
            row = idx // 3
            col_idx = idx % 3
            ax = axes[row, col_idx]
            color = palette[idx]
            ax.fill_between(df['date'], 0, df[col], color=color, alpha=0.3)
            ax.plot(df['date'], df[col], color=color, linewidth=2)
            ax.set_title(title)
            ax.set_ylim(0, 100)
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.charts_dir / f'{inst}_topic_indices_area.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_calendar_heatmaps(self):
        """Create calendar heatmap visualizations (uses sparse data)."""
        if not self.config['charts']['create_calendars']:
            return

        print("\nCreating calendar heatmaps...")
        indices = self.load_indices()

        for inst in ['fed', 'ecb']:
            df = indices[f'{inst}_sparse']
            self._create_policy_calendar(df, inst)
            self._create_topic_calendar(df, inst)
            self._create_market_calendar(df, inst)

        print("  Calendar heatmaps complete")

    def _create_policy_calendar(self, df, inst):
        """Create policy metrics calendar heatmap."""
        inst_name = 'Federal Reserve' if inst == 'fed' else 'European Central Bank'
        variables = [
            'hawkish_dovish_score',
            'uncertainty',
            'forward_guidance_strength',
            'speech_count'
        ]

        years = sorted(df['date'].dt.year.unique())
        n_metrics = len(variables)
        n_rows = n_metrics * len(years)

        fig, axes = plt.subplots(
            nrows=n_rows, ncols=1, figsize=(16, 2.5 * n_rows),
            gridspec_kw={'hspace': 0.4}
        )

        if n_rows == 1:
            axes = [axes]

        fig.suptitle(f'{inst_name} - Policy Metrics (Calendar)', fontsize=16, fontweight='bold', y=0.998)

        ax_idx = 0
        for var in variables:
            metric_title = format_metric_title(var)
            cmap, vcenter, vmin, vmax = get_colormap_settings(var)

            for year in years:
                ax = axes[ax_idx]
                year_df = df[df['date'].dt.year == year].copy()

                if len(year_df) > 0:
                    dates = year_df['date'].tolist()
                    values = year_df[var].tolist()

                    if var in ['uncertainty', 'forward_guidance_strength']:
                        values = [0.01 if v == 0 else v for v in values]

                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"

                    if var == 'speech_count':
                        legend_bins_count = 6
                        legend_labels_custom = 'auto'
                    elif var in ['uncertainty', 'forward_guidance_strength']:
                        legend_bins_count = 5
                        legend_labels_custom = None
                    else:
                        legend_bins_count = 11
                        legend_labels_custom = 'auto'

                    if vcenter is not None:
                        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                        date_series = pd.Series(values, index=dates)
                        full_series = date_series.reindex(full_dates, fill_value=vcenter)

                        dp.calendar(
                            dates=full_series.index.tolist(),
                            values=full_series.tolist(),
                            start_date=start_date,
                            end_date=end_date,
                            cmap=cmap,
                            vcenter=vcenter,
                            vmin=vmin,
                            vmax=vmax,
                            edgecolor='white',
                            edgewidth=0.5,
                            legend=True,
                            legend_bins=legend_bins_count,
                            legend_labels=legend_labels_custom,
                            ax=ax
                        )
                    else:
                        dp.calendar(
                            dates=dates,
                            values=values,
                            start_date=start_date,
                            end_date=end_date,
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            color_for_none='#e8e8e8',
                            edgecolor='white',
                            edgewidth=0.5,
                            legend=True,
                            legend_bins=legend_bins_count,
                            legend_labels=legend_labels_custom,
                            ax=ax
                        )

                ax.text(-4, 3.5, str(year), size=16, rotation=90, color='#666',
                       va='center', ha='center', weight='bold')

                if year == years[0]:
                    ax.text(1.02, 0.5, metric_title, transform=ax.transAxes,
                           size=14, va='center', ha='left', weight='bold', color='#333')

                ax_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.995])
        plt.savefig(self.charts_dir / f'{inst}_policy_metrics_calendar.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_topic_calendar(self, df, inst):
        """Create topic indices calendar heatmap."""
        inst_name = 'Federal Reserve' if inst == 'fed' else 'European Central Bank'
        variables = [
            'topic_inflation',
            'topic_growth',
            'topic_financial_stability',
            'topic_labor_market',
            'topic_international'
        ]

        years = sorted(df['date'].dt.year.unique())
        n_metrics = len(variables)
        n_rows = n_metrics * len(years)

        fig, axes = plt.subplots(
            nrows=n_rows, ncols=1, figsize=(16, 2.5 * n_rows),
            gridspec_kw={'hspace': 0.4}
        )

        if n_rows == 1:
            axes = [axes]

        fig.suptitle(f'{inst_name} - Topic Emphasis (Calendar)', fontsize=16, fontweight='bold', y=0.998)

        ax_idx = 0
        for var in variables:
            metric_title = format_metric_title(var)
            cmap, vcenter, vmin, vmax = get_colormap_settings(var)

            for year in years:
                ax = axes[ax_idx]
                year_df = df[df['date'].dt.year == year].copy()

                if len(year_df) > 0:
                    dates = year_df['date'].tolist()
                    values = year_df[var].tolist()

                    values = [0.01 if v == 0 else v for v in values]

                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"

                    dp.calendar(
                        dates=dates,
                        values=values,
                        start_date=start_date,
                        end_date=end_date,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        color_for_none='#e8e8e8',
                        edgecolor='white',
                        edgewidth=0.5,
                        legend=True,
                        legend_bins=5,
                        legend_labels=None,
                        ax=ax
                    )

                ax.text(-4, 3.5, str(year), size=16, rotation=90, color='#666',
                       va='center', ha='center', weight='bold')

                if year == years[0]:
                    ax.text(1.02, 0.5, metric_title, transform=ax.transAxes,
                           size=14, va='center', ha='left', weight='bold', color='#333')

                ax_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.995])
        plt.savefig(self.charts_dir / f'{inst}_topic_indices_calendar.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_market_calendar(self, df, inst):
        """Create market impact calendar heatmap."""
        inst_name = 'Federal Reserve' if inst == 'fed' else 'European Central Bank'
        variables = [
            'stocks_diffusion_index',
            'bonds_diffusion_index',
            'currency_diffusion_index'
        ]

        years = sorted(df['date'].dt.year.unique())
        n_metrics = len(variables)
        n_rows = n_metrics * len(years)

        fig, axes = plt.subplots(
            nrows=n_rows, ncols=1, figsize=(16, 2.5 * n_rows),
            gridspec_kw={'hspace': 0.4}
        )

        if n_rows == 1:
            axes = [axes]

        fig.suptitle(f'{inst_name} - Market Impact (Calendar)', fontsize=16, fontweight='bold', y=0.998)

        ax_idx = 0
        for var in variables:
            metric_title = format_metric_title(var)
            cmap, vcenter, vmin, vmax = get_colormap_settings(var)

            for year in years:
                ax = axes[ax_idx]
                year_df = df[df['date'].dt.year == year].copy()

                if len(year_df) > 0:
                    dates = year_df['date'].tolist()
                    values = year_df[var].tolist()

                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"

                    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    date_series = pd.Series(values, index=dates)
                    full_series = date_series.reindex(full_dates, fill_value=vcenter)

                    dp.calendar(
                        dates=full_series.index.tolist(),
                        values=full_series.tolist(),
                        start_date=start_date,
                        end_date=end_date,
                        cmap=cmap,
                        vcenter=vcenter,
                        vmin=vmin,
                        vmax=vmax,
                        edgecolor='white',
                        edgewidth=0.5,
                        legend=True,
                        legend_bins=11,
                        legend_labels='auto',
                        ax=ax
                    )

                ax.text(-4, 3.5, str(year), size=16, rotation=90, color='#666',
                       va='center', ha='center', weight='bold')

                if year == years[0]:
                    ax.text(1.02, 0.5, metric_title, transform=ax.transAxes,
                           size=14, va='center', ha='left', weight='bold', color='#333')

                ax_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.995])
        plt.savefig(self.charts_dir / f'{inst}_market_impact_calendar.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_all_charts(self):
        """Create all enabled visualizations."""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        self.create_bar_charts()
        self.create_area_plots()
        self.create_calendar_heatmaps()

        print("\n" + "=" * 70)
        print("VISUALIZATION COMPLETE")
        print("=" * 70)
        print(f"\nCharts saved to: {self.charts_dir}")
