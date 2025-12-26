"""
Step 2: Process Batches and Build Indices

Three modes of operation:
1. Default: Submit all chunks
2. --chunk N: Submit only chunk N
3. --resume: Check status, download completed, resubmit failed

Usage:
  python 02_make_indices.py              # Submit all chunks
  python 02_make_indices.py --chunk 5    # Submit only chunk 5
  python 02_make_indices.py --resume     # Resume from existing batches

Input: data/processed/sample_*.csv
Output:
  - outputs/batch_files/*.jsonl
  - outputs/batch_results/chunk*_results.jsonl
  - outputs/indices/fed_daily_indices.csv
  - outputs/indices/ecb_daily_indices.csv
  - outputs/reports/validation_report.json
  - outputs/reports/batch_info.json
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import utils
from batch_processor import BatchProcessor
from index_builder import IndexBuilder
from output_validator import OutputValidator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process central bank speeches with batch API"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--chunk", type=int, metavar="N", help="Submit only chunk N (e.g., --chunk 5)"
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing batches (check status, download completed, resubmit failed)",
    )

    return parser.parse_args()


def submit_all_chunks(processor, speeches_df, config, batch_info_file):
    """Submit all chunks - default behavior."""
    # Create batch files
    utils.print_section_header("CREATE BATCH FILES")
    batch_files, estimated_input_tokens = processor.create_chunked_batch_files(
        speeches_df
    )

    # Show cost estimate
    total_speeches = len(speeches_df)
    estimated_output_tokens = total_speeches * 500

    cost = utils.calculate_cost(estimated_input_tokens, estimated_output_tokens, config)
    print(f"\nEstimated cost:")
    print(f"  Input tokens: {estimated_input_tokens:,}")
    print(f"  Output tokens: ~{estimated_output_tokens:,}")
    print(f"  Total cost: {utils.format_cost(cost)}")

    # Ask user confirmation
    proceed = input("\nProceed with batch submission? (y/n): ")
    if proceed.lower() != "y":
        print("Aborted")
        return

    # Submit batches
    utils.print_section_header("SUBMIT BATCHES")
    submit_only = (
        input("\nSubmit without waiting for completion? (y/n): ").lower() == "y"
    )

    batch_info = processor.process_all_chunks(batch_files, submit_only=submit_only)

    if submit_only:
        print("\nBatches submitted. Use --resume to check status later.")
        return

    # Continue with processing
    process_results(processor, speeches_df, config)


def submit_single_chunk(processor, speeches_df, config, chunk_num, batch_info_file):
    """Submit only a specific chunk."""
    print(f"\nSINGLE CHUNK MODE: Chunk {chunk_num}")

    # Create all batch files to determine chunk file
    batch_files, _ = processor.create_chunked_batch_files(speeches_df)

    # Validate
    if chunk_num < 1 or chunk_num > len(batch_files):
        print(f"Error: Chunk {chunk_num} not found. Valid: 1-{len(batch_files)}")
        return

    chunk_file = batch_files[chunk_num - 1]
    print(f"Submitting: {chunk_file.name}")

    # Submit
    batch_id = processor.submit_batch(chunk_file)
    print(f"Batch ID: {batch_id}")

    # Load existing batch_info and update
    batch_info = utils.load_json(batch_info_file) if batch_info_file.exists() else {}
    batch_info[chunk_file.name] = {
        "chunk_number": chunk_num,
        "batch_id": batch_id,
        "status": "submitted",
        "submitted_at": datetime.now().isoformat(),
    }
    utils.save_json(batch_info, batch_info_file)

    print(f"\nChunk {chunk_num} submitted. Use --resume to check status later.")


def resubmit_failed_batches(processor, speeches_df, config, batch_info, failed_chunks):
    """
    Resubmit all failed batches automatically.

    Args:
        processor: BatchProcessor instance
        speeches_df: Original speeches DataFrame
        config: Configuration dictionary
        batch_info: Current batch_info dictionary
        failed_chunks: List of chunk names that failed

    Returns:
        Updated batch_info dictionary
    """
    print("\n" + "=" * 70)
    print(f"RESUBMITTING {len(failed_chunks)} FAILED BATCHES")
    print("=" * 70)

    # Ask if they want to wait for completion
    submit_only = (
        input("\nSubmit without waiting for completion? (y/n): ").lower() == "y"
    )

    # Use existing batch files (already created during initial submission)
    print("\nLocating existing batch files...")
    failed_batch_files = []

    for chunk_name in failed_chunks:
        batch_file = processor.batch_files_dir / chunk_name
        if batch_file.exists():
            failed_batch_files.append(batch_file)
            print(f"  Found: {chunk_name}")
        else:
            print(f"  Warning: {chunk_name} not found, will need to recreate")

    # If some files are missing, recreate all batch files
    if len(failed_batch_files) < len(failed_chunks):
        print("\nSome batch files missing. Recreating all batch files...")
        all_batch_files, _ = processor.create_chunked_batch_files(speeches_df)

        # Filter to only failed chunks
        failed_batch_files = []
        for chunk_name in failed_chunks:
            for bf in all_batch_files:
                if bf.name == chunk_name:
                    failed_batch_files.append(bf)
                    break

    print(f"\nResubmitting {len(failed_batch_files)} batch files")

    # Resubmit each failed chunk
    for i, batch_file in enumerate(failed_batch_files, 1):
        print(f"\n{'='*70}")
        print(f"Resubmitting chunk {i}/{len(failed_batch_files)}: {batch_file.name}")
        print(f"{'='*70}")

        try:
            # Submit batch
            batch_id = processor.submit_batch(batch_file)

            # Get batch details
            batch_response = processor.client.batches.retrieve(batch_id)

            # Count requests
            with open(batch_file, "r", encoding="utf-8") as f:
                num_requests = sum(1 for _ in f)

            # Get original chunk number
            original_info = batch_info.get(batch_file.name, {})
            chunk_number = original_info.get("chunk_number", i)

            # Update batch_info
            batch_info[batch_file.name] = {
                "chunk_number": chunk_number,
                "batch_id": batch_id,
                "file_id": batch_response.input_file_id,
                "num_requests": num_requests,
                "status": "submitted",
                "submitted_at": datetime.now().isoformat(),
                "resubmitted": True,
                "previous_batch_id": original_info.get("batch_id"),
            }

            # Save immediately
            processor._save_batch_info(batch_info)
            print(f"  Resubmitted ({num_requests} requests)")

            if not submit_only:
                # Create callback for status updates
                def update_status(new_status):
                    batch_info[batch_file.name]["status"] = new_status
                    batch_info[batch_file.name][
                        "updated_at"
                    ] = datetime.now().isoformat()
                    processor._save_batch_info(batch_info)
                    print(f"  Status updated: {new_status}")

                # Monitor with callback
                status = processor.monitor_batch(
                    batch_id, on_status_change=update_status
                )

                batch_info[batch_file.name]["status"] = status
                batch_info[batch_file.name]["completed_at"] = datetime.now().isoformat()
                processor._save_batch_info(batch_info)

                if status == "completed":
                    # Download results
                    output_file = processor.batch_results_dir / batch_file.name.replace(
                        "_input.jsonl", "_results.jsonl"
                    )
                    processor.download_results(batch_id, output_file)
                    batch_info[batch_file.name]["output_file"] = str(output_file)
                    processor._save_batch_info(batch_info)
                    print(f"  Downloaded results")

        except Exception as e:
            print(f"  X Error resubmitting: {e}")
            batch_info[batch_file.name]["resubmit_error"] = str(e)
            processor._save_batch_info(batch_info)

    return batch_info


def resume_workflow(processor, speeches_df, config, batch_info_file):
    """Resume from existing batches."""
    print("\nRESUME MODE")

    if not batch_info_file.exists():
        print("No batch_info.json found.")
        print("Options:")
        print("  1. Run without --resume to start fresh")
        print("  2. Check https://platform.openai.com/batches for batch IDs")
        return

    # Load existing info
    batch_info = utils.load_json(batch_info_file)

    # Find all batch files on disk
    all_batch_files = sorted(processor.batch_files_dir.glob("chunk*_input.jsonl"))
    print(f"\nFound {len(all_batch_files)} batch files on disk")

    # Identify chunks that exist as files but not in batch_info.json
    tracked_chunks = {k for k in batch_info.keys() if not k.startswith("_")}
    missing_files = [f for f in all_batch_files if f.name not in tracked_chunks]

    # Handle missing chunks (not yet submitted)
    if missing_files:
        print(f"\n{'='*70}")
        print(f"WARNING: {len(missing_files)} chunks not yet submitted")
        print(f"{'='*70}")
        print("\nThese batch files exist but are not tracked in batch_info.json:")
        for f in missing_files:
            chunk_num = int(f.name.replace("chunk", "").replace("_input.jsonl", ""))
            with open(f, "r", encoding="utf-8") as file:
                num_requests = sum(1 for _ in file)
            print(f"  - {f.name} (chunk {chunk_num}, {num_requests} requests)")

        print("\nThis typically happens when:")
        print("  - Billing limit was hit during submission")
        print("  - Script crashed or was interrupted (Ctrl+C)")
        print("  - Network failure during batch submission")

        submit = input(f"\nSubmit {len(missing_files)} missing chunks now? (y/n): ")

        if submit.lower() == "y":
            submit_only = (
                input("\nSubmit without waiting for completion? (y/n): ").lower() == "y"
            )

            print(f"\n{'='*70}")
            print(f"SUBMITTING {len(missing_files)} MISSING CHUNKS")
            print(f"{'='*70}")

            for i, batch_file in enumerate(missing_files, 1):
                print(f"\n{'='*70}")
                print(
                    f"Submitting missing chunk {i}/{len(missing_files)}: {batch_file.name}"
                )
                print(f"{'='*70}")

                try:
                    # Submit batch
                    batch_id = processor.submit_batch(batch_file)

                    # Get batch details
                    batch_response = processor.client.batches.retrieve(batch_id)

                    # Count requests
                    with open(batch_file, "r", encoding="utf-8") as f:
                        num_requests = sum(1 for _ in f)

                    # Extract chunk number from filename
                    chunk_num = int(
                        batch_file.name.replace("chunk", "").replace("_input.jsonl", "")
                    )

                    # Add to batch_info
                    batch_info[batch_file.name] = {
                        "chunk_number": chunk_num,
                        "batch_id": batch_id,
                        "file_id": batch_response.input_file_id,
                        "num_requests": num_requests,
                        "status": "submitted",
                        "submitted_at": datetime.now().isoformat(),
                    }

                    # Save immediately
                    processor._save_batch_info(batch_info)
                    print(f"  Submitted ({num_requests} requests)")

                    if not submit_only:
                        # Monitor with callback
                        def update_status(new_status):
                            batch_info[batch_file.name]["status"] = new_status
                            batch_info[batch_file.name][
                                "updated_at"
                            ] = datetime.now().isoformat()
                            processor._save_batch_info(batch_info)
                            print(f"  Status updated: {new_status}")

                        status = processor.monitor_batch(
                            batch_id, on_status_change=update_status
                        )

                        batch_info[batch_file.name]["status"] = status
                        batch_info[batch_file.name][
                            "completed_at"
                        ] = datetime.now().isoformat()
                        processor._save_batch_info(batch_info)

                        if status == "completed":
                            # Download results
                            output_file = (
                                processor.batch_results_dir
                                / batch_file.name.replace(
                                    "_input.jsonl", "_results.jsonl"
                                )
                            )
                            processor.download_results(batch_id, output_file)
                            batch_info[batch_file.name]["output_file"] = str(
                                output_file
                            )
                            processor._save_batch_info(batch_info)
                            print(f"  Downloaded results")

                except Exception as e:
                    print(f"  X Error: {e}")
                    chunk_num = int(
                        batch_file.name.replace("chunk", "").replace("_input.jsonl", "")
                    )
                    batch_info[batch_file.name] = {
                        "chunk_number": chunk_num,
                        "status": "error",
                        "error": str(e),
                        "failed_at": datetime.now().isoformat(),
                    }
                    processor._save_batch_info(batch_info)

            print(f"\nFinished submitting {len(missing_files)} missing chunks")

            # Reload batch_info after submissions
            batch_info = utils.load_json(batch_info_file)

    # Display metadata if available
    if "_metadata" in batch_info:
        meta = batch_info["_metadata"]
        print(f"\nBatch submission metadata:")
        print(f"  Total chunks: {meta.get('total_chunks', 'unknown')}")
        print(f"  Total speeches: {meta.get('total_speeches', 'unknown')}")
        print(f"  Started: {meta.get('started_at', 'unknown')}")
        if "completed_at" in meta:
            print(f"  Completed: {meta['completed_at']}")

    # Count non-metadata entries
    chunks = {k: v for k, v in batch_info.items() if not k.startswith("_")}
    print(f"\nFound {len(chunks)} batches in batch_info.json")

    # Check current status
    print("\nChecking status with OpenAI...")
    print(f"{'Chunk':<8} {'Requests':<10} {'Status':<15} {'Batch ID':<40}")
    print("-" * 80)

    updated_info = {}

    for chunk_name, info in chunks.items():
        batch_id = info["batch_id"]
        chunk_num = info.get("chunk_number", "?")
        num_requests = info.get("num_requests", "?")

        try:
            batch = processor.client.batches.retrieve(batch_id)
            updated_info[chunk_name] = {
                **info,
                "status": batch.status,
                "checked_at": datetime.now().isoformat(),
            }
            print(
                f"{chunk_num:<8} {num_requests:<10} {batch.status:<15} {batch_id:<40}"
            )
        except Exception as e:
            print(f"{chunk_num:<8} {num_requests:<10} {'error':<15} {batch_id:<40}")
            updated_info[chunk_name] = info

    # Keep metadata
    if "_metadata" in batch_info:
        updated_info["_metadata"] = batch_info["_metadata"]

    # Categorize
    completed = [
        k
        for k, v in updated_info.items()
        if not k.startswith("_") and v["status"] == "completed"
    ]
    in_progress = [
        k
        for k, v in updated_info.items()
        if not k.startswith("_")
        and v["status"] in ["validating", "in_progress", "finalizing"]
    ]
    failed = [
        k
        for k, v in updated_info.items()
        if not k.startswith("_") and v["status"] in ["failed", "expired", "cancelled"]
    ]

    print(f"\nStatus summary:")
    print(f"  Completed: {len(completed)}")
    print(f"  In progress: {len(in_progress)}")
    print(f"  Failed: {len(failed)}")

    # Download completed
    if completed:
        print("\nDownloading completed batches...")
        for chunk_name in completed:
            info = updated_info[chunk_name]
            if "output_file" not in info:
                output_file = processor.batch_results_dir / chunk_name.replace(
                    "_input.jsonl", "_results.jsonl"
                )
                if not output_file.exists():
                    processor.download_results(info["batch_id"], output_file)
                    updated_info[chunk_name]["output_file"] = str(output_file)
                    print(f"  Downloaded: {chunk_name}")
                else:
                    updated_info[chunk_name]["output_file"] = str(output_file)
                    print(f"  Already exists: {chunk_name}")

    # Save updated info
    utils.save_json(updated_info, batch_info_file)

    # Handle different scenarios
    total_batches = len(chunks)

    if len(completed) == total_batches:
        # All complete - proceed
        proceed = input("\nAll batches complete. Process results? (y/n): ")
        if proceed.lower() == "y":
            process_results(processor, speeches_df, config)

    elif failed and not in_progress:
        # Only failed batches left (no in-progress)
        print(f"\n{len(failed)} batches failed (no batches in progress):")
        for chunk_name in failed:
            info = updated_info[chunk_name]
            print(f"  - {chunk_name} (Batch ID: {info['batch_id']})")

        resubmit = input(f"\nResubmit all {len(failed)} failed batches? (y/n): ")

        if resubmit.lower() == "y":
            updated_info = resubmit_failed_batches(
                processor, speeches_df, config, updated_info, failed
            )

            # Save updated info
            utils.save_json(updated_info, batch_info_file)

            # Check results after resubmission
            all_chunks = {
                k: v for k, v in updated_info.items() if not k.startswith("_")
            }
            newly_completed = [
                k for k, v in all_chunks.items() if v["status"] == "completed"
            ]
            still_failed = [
                k
                for k, v in all_chunks.items()
                if v["status"] in ["failed", "expired", "cancelled"]
            ]

            print("\n" + "=" * 70)
            print("RESUBMISSION COMPLETE")
            print("=" * 70)
            print(f"\nTotal batches: {len(all_chunks)}")
            print(f"Completed: {len(newly_completed)}")
            print(f"Still failed: {len(still_failed)}")

            if len(newly_completed) == len(all_chunks):
                proceed = input("\nAll batches complete! Process results? (y/n): ")
                if proceed.lower() == "y":
                    process_results(processor, speeches_df, config)
            elif still_failed:
                print(
                    f"\n{len(still_failed)} batches still failed. Run --resume again or resubmit manually."
                )
        else:
            print("\nSkipped resubmission. Resubmit manually with --chunk N")

    elif in_progress and failed:
        # Both in-progress and failed
        print(f"\nMixed status:")
        print(f"  {len(in_progress)} batches still processing")
        print(f"  {len(failed)} batches failed:")
        for chunk_name in failed:
            info = updated_info[chunk_name]
            print(f"    - {chunk_name} (Batch ID: {info['batch_id']})")

        print("\nWhat would you like to do?")
        print("  1. Wait (run --resume again later when in-progress batches complete)")
        print(
            "  2. Resubmit failed batches now (in-progress batches continue separately)"
        )
        choice = input("Choice (1/2): ")

        if choice == "2":
            updated_info = resubmit_failed_batches(
                processor, speeches_df, config, updated_info, failed
            )

            # Save updated info
            utils.save_json(updated_info, batch_info_file)

            print(f"\n{len(failed)} failed batches resubmitted.")
            print(f"{len(in_progress)} batches still processing from before.")
            print("Run --resume again later to check all statuses.")
        else:
            print(f"\n{len(in_progress)} batches still processing.")
            print(f"{len(failed)} batches failed.")
            print("Run --resume again later.")

    elif in_progress:
        # Only in-progress
        print(
            f"\n{len(in_progress)} batches still processing. Run --resume again later."
        )


def process_results(processor, speeches_df, config):
    """Combine results, validate, build indices."""
    start_year = config["date_range"]["start"][:4]
    end_year = config["date_range"]["end"][:4]

    # Combine
    utils.print_section_header("COMBINE RESULTS")
    combined_results = processor.combine_chunk_results(speeches_df)

    # Save
    results_file = f"{config['directories']['processed_data']}/sentiment_results_{start_year}_{end_year}.csv"
    combined_results.to_csv(results_file, index=False)
    print(f"\nSaved: {results_file}")

    # Validate
    utils.print_section_header("VALIDATE OUTPUTS")
    validator = OutputValidator()
    validation_results = validator.validate_batch_results(combined_results)
    validator.print_validation_report(validation_results)

    # Save validation report
    utils.save_json(
        validation_results, f"{config['directories']['reports']}/validation_report.json"
    )

    # Build indices
    utils.print_section_header("BUILD DAILY INDICES")
    builder = IndexBuilder(config)
    builder.build_indices(combined_results)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


def main():
    """Main execution function."""
    args = parse_args()

    utils.print_section_header("STEP 2: BATCH PROCESSING & INDEX BUILDING")

    # Load configuration
    print("\nLoading configuration...")
    config = utils.load_config()
    utils.ensure_directories(config)

    # Define paths
    batch_info_file = Path(config["directories"]["reports"]) / "batch_info.json"
    date_range = config["date_range"]
    start_year = date_range["start"][:4]
    end_year = date_range["end"][:4]
    speeches_file = (
        f"{config['directories']['processed_data']}/sample_{start_year}_{end_year}.csv"
    )

    # Load speeches
    print(f"\nLoading speeches from: {speeches_file}")
    speeches_df = pd.read_csv(speeches_file, parse_dates=["date"])
    print(f"Total speeches: {len(speeches_df)}")

    # Initialize processor with batch_info_file path
    processor = BatchProcessor(config, batch_info_file=batch_info_file)

    # Route to appropriate workflow
    if args.resume:
        resume_workflow(processor, speeches_df, config, batch_info_file)
    elif args.chunk is not None:
        submit_single_chunk(processor, speeches_df, config, args.chunk, batch_info_file)
    else:
        submit_all_chunks(processor, speeches_df, config, batch_info_file)


if __name__ == "__main__":
    main()
