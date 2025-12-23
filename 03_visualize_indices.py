"""
Step 3: Create Visualizations

Creates all visualizations from daily indices:
- Bar charts (sparse data)
- Area plots (forward-filled data)
- Calendar heatmaps (sparse data)

Usage: python 03_visualize_indices.py
Input: outputs/indices/*.csv
Output: outputs/charts/*.png
"""

import utils
from visualizer import Visualizer


def main():
    """Main execution function."""
    utils.print_section_header("STEP 3: CREATE VISUALIZATIONS")

    # Load configuration
    print("\nLoading configuration...")
    config = utils.load_config()
    utils.ensure_directories(config)

    # Initialize visualizer
    viz = Visualizer(config)

    # Create all charts
    viz.create_all_charts()

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)
    print(f"\nCharts saved to: {config['directories']['charts']}/")

    # List created files
    from pathlib import Path
    charts_dir = Path(config['directories']['charts'])
    chart_files = sorted(charts_dir.glob("*.png"))

    if chart_files:
        print(f"\nCreated {len(chart_files)} chart files:")
        for f in chart_files:
            print(f"  - {f.name}")
    else:
        print("\nNo chart files were created.")


if __name__ == "__main__":
    main()
