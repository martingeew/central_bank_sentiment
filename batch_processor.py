"""
Batch processor for OpenAI Batch API.

Merged from batch_builder and batch_processor.
Handles JSONL creation, chunking, submission, and monitoring.
"""

import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
import utils


class BatchProcessor:
    """
    Manages OpenAI Batch API operations with automatic chunking.
    """

    def __init__(self, config: Dict[str, Any], batch_info_file: Path = None):
        """
        Initialize batch processor.

        Args:
            config: Configuration dictionary
            batch_info_file: Optional path to batch_info.json for incremental saves
        """
        self.config = config
        self.api_key = config['api_keys']['openai']
        self.client = OpenAI(api_key=self.api_key)

        self.batch_files_dir = Path(config['directories']['batch_files'])
        self.batch_results_dir = Path(config['directories']['batch_results'])
        self.batch_info_file = batch_info_file

    def _save_batch_info(self, batch_info: Dict[str, Any]):
        """Save batch info to file incrementally."""
        if self.batch_info_file:
            utils.save_json(batch_info, self.batch_info_file)

    def create_batch_request(self, speech_id: str, speech_text: str,
                             speaker: str, institution: str, date: str) -> Dict[str, Any]:
        """
        Create single batch API request.

        Args:
            speech_id: Unique identifier
            speech_text: Speech content
            speaker: Speaker name
            institution: United States or Euro area
            date: Speech date

        Returns:
            Batch request dictionary
        """
        prompt = utils.get_sentiment_prompt(speech_text, speaker, institution, date)

        request = {
            "custom_id": speech_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.config['model']['name'],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in central bank communication analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "response_format": {"type": self.config['model']['response_format']},
                "temperature": self.config['model']['temperature']
            }
        }

        return request

    def create_chunked_batch_files(self, speeches_df: pd.DataFrame) -> tuple[List[Path], int]:
        """
        Create batch files with automatic chunking based on token limits.

        Uses token estimation to ensure each chunk stays under the token limit.
        Handles variable speech lengths by dynamically grouping speeches.

        Args:
            speeches_df: DataFrame with speeches

        Returns:
            Tuple of (batch file paths, total estimated input tokens)
        """
        max_tokens_per_chunk = self.config['chunking']['max_tokens_per_chunk']
        total_speeches = len(speeches_df)

        print(f"\nCreating batch chunks with token-based splitting...")
        print(f"Total speeches: {total_speeches}")
        print(f"Max tokens per chunk: {max_tokens_per_chunk:,}")

        # Build all requests first and estimate tokens
        all_requests = []
        print("\nEstimating tokens for each request...")

        for idx, row in speeches_df.iterrows():
            request = self.create_batch_request(
                speech_id=row['speech_id'],
                speech_text=row['text'],
                speaker=row['author'],
                institution=row['country'],
                date=str(row['date'].date())
            )

            # Estimate tokens for this request
            request_str = json.dumps(request)
            request_tokens = utils.estimate_tokens(request_str)

            all_requests.append({
                'request': request,
                'tokens': request_tokens
            })

        # Split into chunks based on token limits
        chunks = []
        current_chunk = []
        current_tokens = 0

        for req_info in all_requests:
            request_tokens = req_info['tokens']

            # If adding this request would exceed limit, start new chunk
            if current_tokens + request_tokens > max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [req_info]
                current_tokens = request_tokens
            else:
                current_chunk.append(req_info)
                current_tokens += request_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        print(f"\nSplit into {len(chunks)} chunks:")

        # Write chunks to files
        batch_files = []
        total_input_tokens = 0

        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_file = self.batch_files_dir / f"chunk{chunk_idx:02d}_input.jsonl"

            # Calculate chunk statistics
            chunk_tokens = sum(req['tokens'] for req in chunk)
            chunk_speeches = len(chunk)
            total_input_tokens += chunk_tokens

            # Write chunk to JSONL file
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for req_info in chunk:
                    f.write(json.dumps(req_info['request']) + '\n')

            print(f"  Chunk {chunk_idx:2d}: {chunk_speeches:3d} speeches, ~{chunk_tokens:,} tokens")
            batch_files.append(chunk_file)

        print(f"\nTotal estimated input tokens: {total_input_tokens:,}")

        return batch_files, total_input_tokens

    def submit_batch(self, batch_file: Path) -> str:
        """
        Upload file and submit batch job.

        Args:
            batch_file: Path to JSONL batch file

        Returns:
            Batch job ID
        """
        print(f"\nSubmitting batch: {batch_file.name}")

        # Upload file
        with open(batch_file, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose='batch'
            )

        file_id = file_response.id
        print(f"  Uploaded file ID: {file_id}")

        # Create batch job
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        batch_id = batch_response.id
        print(f"  Batch job ID: {batch_id}")
        print(f"  Status: {batch_response.status}")

        return batch_id

    def monitor_batch(self, batch_id: str, check_interval: int = 60,
                      on_status_change=None) -> str:
        """
        Monitor batch job until completion with optional status callback.

        Args:
            batch_id: Batch job ID
            check_interval: Seconds between status checks
            on_status_change: Optional callback function(status) called on status change

        Returns:
            Final status (completed, failed, etc.)
        """
        print(f"\nMonitoring batch: {batch_id}")
        last_status = None

        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            status = batch_status.status

            # Call callback if status changed
            if on_status_change and status != last_status:
                on_status_change(status)
                last_status = status

            if status == 'completed':
                print(f"  Batch completed successfully")
                return status
            elif status in ['failed', 'expired', 'cancelled']:
                print(f"  Batch {status}")
                return status
            else:
                completed = batch_status.request_counts.completed
                total = batch_status.request_counts.total
                print(f"  Status: {status} - {completed}/{total} requests completed")
                time.sleep(check_interval)

    def download_results(self, batch_id: str, output_file: Path) -> Path:
        """
        Download batch results.

        Args:
            batch_id: Batch job ID
            output_file: Where to save results

        Returns:
            Path to downloaded file
        """
        batch_status = self.client.batches.retrieve(batch_id)

        if batch_status.output_file_id is None:
            raise ValueError(f"Batch {batch_id} has no output file")

        print(f"\nDownloading results: {output_file.name}")

        # Download file content
        file_response = self.client.files.content(batch_status.output_file_id)
        content = file_response.read()

        # Save to file
        with open(output_file, 'wb') as f:
            f.write(content)

        print(f"  Saved to: {output_file}")
        return output_file

    def parse_results(self, results_file: Path) -> pd.DataFrame:
        """
        Parse batch results JSONL into DataFrame.

        Args:
            results_file: Path to results JSONL

        Returns:
            DataFrame with parsed sentiment scores
        """
        print(f"\nParsing results: {results_file.name}")

        results = []

        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    response = json.loads(line)
                    custom_id = response['custom_id']

                    # Extract JSON from response
                    content = response['response']['body']['choices'][0]['message']['content']
                    sentiment_data = json.loads(content)

                    # Flatten into row
                    row = {
                        'speech_id': custom_id,
                        'hawkish_dovish_score': sentiment_data['hawkish_dovish_score'],
                        'uncertainty': sentiment_data['uncertainty'],
                        'forward_guidance_strength': sentiment_data['forward_guidance_strength'],
                        'topic_inflation': sentiment_data['topics']['inflation'],
                        'topic_growth': sentiment_data['topics']['growth'],
                        'topic_financial_stability': sentiment_data['topics']['financial_stability'],
                        'topic_labor_market': sentiment_data['topics']['labor_market'],
                        'topic_international': sentiment_data['topics']['international'],
                        'market_impact_stocks': sentiment_data['market_impact']['stocks'],
                        'market_impact_bonds': sentiment_data['market_impact']['bonds'],
                        'market_impact_currency': sentiment_data['market_impact']['currency'],
                        'market_impact_reasoning': sentiment_data['market_impact'].get('reasoning', ''),
                        'key_sentences': '|'.join(sentiment_data.get('key_sentences', [])),
                        'summary': sentiment_data.get('summary', '')
                    }

                    results.append(row)

                except Exception as e:
                    print(f"  Error parsing {custom_id}: {e}")

        df = pd.DataFrame(results)
        print(f"  Parsed {len(df)} speeches successfully")

        return df

    def process_all_chunks(self, batch_files: List[Path],
                           submit_only: bool = False) -> Dict[str, Any]:
        """
        Process all batch chunks with incremental saves and enhanced metadata.

        Args:
            batch_files: List of batch file paths
            submit_only: If True, only submit without waiting

        Returns:
            Dictionary mapping chunk files to batch info
        """
        # Load existing or create new
        if self.batch_info_file and self.batch_info_file.exists():
            batch_info = utils.load_json(self.batch_info_file)
            print(f"\nLoaded existing batch_info with {len(batch_info)} entries")
        else:
            batch_info = {}

        # Create initial file with metadata
        if self.batch_info_file:
            # Calculate total requests
            total_requests = sum(
                sum(1 for _ in open(bf, 'r', encoding='utf-8'))
                for bf in batch_files
            )

            batch_info['_metadata'] = {
                'total_chunks': len(batch_files),
                'total_speeches': total_requests,
                'started_at': datetime.now().isoformat(),
                'status': 'submitting'
            }
            self._save_batch_info(batch_info)
            print(f"Created batch_info file: {self.batch_info_file}")

        for i, batch_file in enumerate(batch_files, 1):
            print(f"\n{'='*70}")
            print(f"Processing chunk {i}/{len(batch_files)}")
            print(f"{'='*70}")

            # Skip if already submitted and completed/in_progress
            if batch_file.name in batch_info:
                existing_status = batch_info[batch_file.name].get('status')
                if existing_status in ['completed', 'in_progress']:
                    print(f"  Skipping {batch_file.name} (already {existing_status})")
                    continue

            try:
                # Submit batch
                batch_id = self.submit_batch(batch_file)

                # Get batch details for metadata
                batch_response = self.client.batches.retrieve(batch_id)

                # Count requests in chunk file
                with open(batch_file, 'r', encoding='utf-8') as f:
                    num_requests = sum(1 for _ in f)

                # Save comprehensive metadata
                batch_info[batch_file.name] = {
                    'chunk_number': i,
                    'batch_id': batch_id,
                    'file_id': batch_response.input_file_id,
                    'num_requests': num_requests,
                    'status': 'submitted',
                    'submitted_at': datetime.now().isoformat()
                }

                # SAVE IMMEDIATELY
                self._save_batch_info(batch_info)
                print(f"  ✓ Saved batch info ({num_requests} requests)")

                if not submit_only:
                    # Create callback to save on status changes
                    def update_status(new_status):
                        batch_info[batch_file.name]['status'] = new_status
                        batch_info[batch_file.name]['updated_at'] = datetime.now().isoformat()
                        self._save_batch_info(batch_info)
                        print(f"  ✓ Status updated: {new_status}")

                    # Monitor with callback
                    status = self.monitor_batch(batch_id, on_status_change=update_status)

                    batch_info[batch_file.name]['status'] = status
                    batch_info[batch_file.name]['completed_at'] = datetime.now().isoformat()
                    self._save_batch_info(batch_info)

                    if status == 'completed':
                        # Download results
                        output_file = self.batch_results_dir / batch_file.name.replace('_input.jsonl', '_results.jsonl')
                        self.download_results(batch_id, output_file)
                        batch_info[batch_file.name]['output_file'] = str(output_file)
                        self._save_batch_info(batch_info)
                        print(f"  ✓ Downloaded results")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                batch_info[batch_file.name] = {
                    'status': 'error',
                    'error': str(e),
                    'failed_at': datetime.now().isoformat()
                }
                self._save_batch_info(batch_info)

        # Update final metadata
        if self.batch_info_file:
            batch_info['_metadata']['completed_at'] = datetime.now().isoformat()
            batch_info['_metadata']['status'] = 'completed'
            self._save_batch_info(batch_info)

        return batch_info

    def combine_chunk_results(self, speeches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all chunk results into single DataFrame.

        Args:
            speeches_df: Original speeches DataFrame

        Returns:
            Combined DataFrame with sentiment scores
        """
        print("\nCombining results from all chunks...")

        # Find all result files
        result_files = sorted(self.batch_results_dir.glob("chunk*_results.jsonl"))

        if not result_files:
            raise ValueError("No result files found")

        print(f"Found {len(result_files)} result files")

        # Parse each file and combine
        all_results = []

        for result_file in result_files:
            parsed = self.parse_results(result_file)
            all_results.append(parsed)

        # Combine all chunks
        combined = pd.concat(all_results, ignore_index=True)
        print(f"\nTotal speeches with results: {len(combined)}")

        # Merge with original speeches to add metadata
        merged = speeches_df.merge(combined, on='speech_id', how='inner')

        print(f"Merged with speech metadata: {len(merged)} speeches")

        return merged
