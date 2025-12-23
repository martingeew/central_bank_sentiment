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
from openai import OpenAI
from tqdm import tqdm
import utils


class BatchProcessor:
    """
    Manages OpenAI Batch API operations with automatic chunking.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize batch processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config['api_keys']['openai']
        self.client = OpenAI(api_key=self.api_key)

        self.batch_files_dir = Path(config['directories']['batch_files'])
        self.batch_results_dir = Path(config['directories']['batch_results'])

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

    def create_chunked_batch_files(self, speeches_df: pd.DataFrame) -> List[Path]:
        """
        Create batch files with automatic chunking to stay under token limit.

        Args:
            speeches_df: DataFrame with speeches

        Returns:
            List of batch file paths
        """
        max_per_chunk = self.config['chunking']['max_speeches_per_chunk']
        total_speeches = len(speeches_df)
        num_chunks = (total_speeches + max_per_chunk - 1) // max_per_chunk

        print(f"\nCreating {num_chunks} batch chunks ({max_per_chunk} speeches per chunk)")
        print(f"Total speeches: {total_speeches}")

        batch_files = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * max_per_chunk
            end_idx = min((chunk_idx + 1) * max_per_chunk, total_speeches)
            chunk_speeches = speeches_df.iloc[start_idx:end_idx]

            # Create JSONL file for this chunk
            chunk_file = self.batch_files_dir / f"chunk{chunk_idx+1:02d}_input.jsonl"
            requests = []

            for idx, row in chunk_speeches.iterrows():
                request = self.create_batch_request(
                    speech_id=row['speech_id'],
                    speech_text=row['content'],
                    speaker=row['author'],
                    institution=row['country'],
                    date=str(row['date'].date())
                )
                requests.append(request)

            # Write JSONL file
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for request in requests:
                    f.write(json.dumps(request) + '\n')

            print(f"  Created chunk {chunk_idx+1}/{num_chunks}: {chunk_file.name} ({len(chunk_speeches)} speeches)")
            batch_files.append(chunk_file)

        return batch_files

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

    def monitor_batch(self, batch_id: str, check_interval: int = 60) -> str:
        """
        Monitor batch job until completion.

        Args:
            batch_id: Batch job ID
            check_interval: Seconds between status checks

        Returns:
            Status (completed, failed, etc.)
        """
        print(f"\nMonitoring batch: {batch_id}")

        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            status = batch_status.status

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
        Process all batch chunks sequentially.

        Args:
            batch_files: List of batch file paths
            submit_only: If True, only submit without waiting

        Returns:
            Dictionary mapping chunk files to batch IDs
        """
        batch_info = {}

        for i, batch_file in enumerate(batch_files, 1):
            print(f"\n{'='*70}")
            print(f"Processing chunk {i}/{len(batch_files)}")
            print(f"{'='*70}")

            # Submit batch
            batch_id = self.submit_batch(batch_file)
            batch_info[batch_file.name] = {'batch_id': batch_id, 'status': 'submitted'}

            if not submit_only:
                # Monitor until completion
                status = self.monitor_batch(batch_id)
                batch_info[batch_file.name]['status'] = status

                if status == 'completed':
                    # Download results
                    output_file = self.batch_results_dir / batch_file.name.replace('_input.jsonl', '_results.jsonl')
                    self.download_results(batch_id, output_file)
                    batch_info[batch_file.name]['output_file'] = str(output_file)

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
