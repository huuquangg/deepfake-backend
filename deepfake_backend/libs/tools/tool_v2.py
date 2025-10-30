import os
import csv
import asyncio
import aiohttp
import pandas as pd
from typing import List, Tuple, Optional

# ---------------- CONFIG ----------------
FOLDER_PATH = "/home/huuquangdang/huu.quang.dang/thesis/Dataset/celeb_df_crop/real"
OUTPUT_CSV = "merged_vectors_real.csv"

ANALYZE_API = "http://127.0.0.1:8000/api/detection/analyze-frame"

# Optimization settings
BATCH_SIZE = 30  # Process 30 images in parallel
TIMEOUT = 60  # Timeout per request in seconds
MAX_RETRIES = 2  # Retry failed requests
# ----------------------------------------

async def process_image(session, file_path, filename, retry_count=0) -> Tuple[str, Optional[List]]:
    """Process a single image: analyze-frame and extract CSV data."""
    try:
        # Call analyze-frame API with timeout
        with open(file_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=filename, content_type='image/jpeg')
            
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            async with session.post(ANALYZE_API, data=data, timeout=timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[ERROR] Analyze-frame failed for {filename}: {error_text}")
                    
                    # Retry on failure
                    if retry_count < MAX_RETRIES:
                        print(f"[RETRY] Retrying {filename} (attempt {retry_count + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(1)  # Brief delay before retry
                        return await process_image(session, file_path, filename, retry_count + 1)
                    return filename, None
                    
                result = await resp.json()
                
                # Get the output directory and read the CSV file
                output_dir = result.get("output_dir")
                if not output_dir:
                    print(f"[ERROR] No output_dir in response for {filename}")
                    return filename, None
                
                # Find the CSV file in the output directory
                csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
                if not csv_files:
                    print(f"[ERROR] No CSV file found in {output_dir}")
                    return filename, None
                
                # Read the CSV file (take the first one)
                csv_path = os.path.join(output_dir, csv_files[0])
                df = pd.read_csv(csv_path)
                
                # Extract the data row (assuming only 1 row of data)
                if len(df) > 0:
                    data_row = df.iloc[0].tolist()
                    return filename, data_row
                else:
                    print(f"[ERROR] Empty CSV for {filename}")
                    return filename, None

    except asyncio.TimeoutError:
        print(f"[TIMEOUT] {filename} - Request timed out after {TIMEOUT}s")
        if retry_count < MAX_RETRIES:
            print(f"[RETRY] Retrying {filename} (attempt {retry_count + 1}/{MAX_RETRIES})")
            return await process_image(session, file_path, filename, retry_count + 1)
        return filename, None
    except Exception as e:
        print(f"[EXCEPTION] {filename}: {e}")
        if retry_count < MAX_RETRIES:
            print(f"[RETRY] Retrying {filename} (attempt {retry_count + 1}/{MAX_RETRIES})")
            await asyncio.sleep(1)
            return await process_image(session, file_path, filename, retry_count + 1)
        return filename, None


async def process_batch(session, batch, batch_num, total_batches) -> List[Tuple[str, Optional[List]]]:
    """Process a batch of images in parallel."""
    print(f"\n{'='*60}")
    print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} images)")
    print(f"{'='*60}")
    
    tasks = []
    for filename in batch:
        file_path = os.path.join(FOLDER_PATH, filename)
        tasks.append(process_image(session, file_path, filename))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions in results
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"[ERROR] Batch processing exception: {result}")
            processed_results.append(("unknown", None))
        else:
            processed_results.append(result)
    
    return processed_results


async def main():
    # Get sorted list of all JPG images
    files_list = sorted(f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".jpg"))
    total_files = len(files_list)
    
    # Split into batches
    batches = [files_list[i:i + BATCH_SIZE] for i in range(0, len(files_list), BATCH_SIZE)]
    total_batches = len(batches)
    
    print(f"{'='*60}")
    print(f"Found {total_files} images to process")
    print(f"Processing in {total_batches} batches of {BATCH_SIZE} images each")
    print(f"{'='*60}\n")

    success_count = 0
    failed_count = 0
    failed_images = []  # Track failed images for retry

    # Open CSV for writing
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Use connector with connection pooling for better performance
        connector = aiohttp.TCPConnector(limit=BATCH_SIZE, limit_per_host=BATCH_SIZE)
        timeout = aiohttp.ClientTimeout(total=None)  # No total timeout, handled per request
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # First pass: Process all images
            for batch_num, batch in enumerate(batches, start=1):
                results = await process_batch(session, batch, batch_num, total_batches)
                
                # Write results
                for fname, data_row in results:
                    if data_row:
                        row_with_filename = [fname] + data_row
                        writer.writerow(row_with_filename)
                        success_count += 1
                        print(f"[✓] {fname} - Success")
                    else:
                        failed_count += 1
                        failed_images.append(fname)
                        # Write failed marker
                        writer.writerow([fname, False])
                        print(f"[✗] {fname} - Failed (marked for retry)")
                
                # Flush after each batch to save progress
                csvfile.flush()
                
                progress = (batch_num / total_batches) * 100
                print(f"\n[PROGRESS] {batch_num}/{total_batches} batches ({progress:.1f}%) | Success: {success_count} | Failed: {failed_count}\n")
                
                # Small delay between batches to avoid overwhelming the server
                if batch_num < total_batches:
                    await asyncio.sleep(0.5)

            # Second pass: Retry failed images
            if failed_images:
                print(f"\n{'='*60}")
                print(f"RETRY PHASE: Retrying {len(failed_images)} failed images")
                print(f"{'='*60}\n")
                
                retry_batches = [failed_images[i:i + BATCH_SIZE] for i in range(0, len(failed_images), BATCH_SIZE)]
                retry_total_batches = len(retry_batches)
                retry_success = 0
                retry_failed = 0
                
                for retry_batch_num, retry_batch in enumerate(retry_batches, start=1):
                    print(f"\n[RETRY BATCH {retry_batch_num}/{retry_total_batches}]")
                    results = await process_batch(session, retry_batch, retry_batch_num, retry_total_batches)
                    
                    # Write retry results
                    for fname, data_row in results:
                        if data_row:
                            row_with_filename = [fname] + data_row
                            writer.writerow(row_with_filename)
                            retry_success += 1
                            success_count += 1
                            failed_count -= 1
                            print(f"[✓] {fname} - Retry Success")
                        else:
                            retry_failed += 1
                            print(f"[✗] {fname} - Retry Failed")
                    
                    # Flush after each retry batch
                    csvfile.flush()
                    
                    if retry_batch_num < retry_total_batches:
                        await asyncio.sleep(0.5)
                
                print(f"\n[RETRY SUMMARY] Success: {retry_success} | Still Failed: {retry_failed}")

    print(f"\n{'='*60}")
    print(f"All done! Results saved to {OUTPUT_CSV}")
    print(f"Total: {total_files} | Success: {success_count} | Failed: {failed_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
