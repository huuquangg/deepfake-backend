import csv
import os
import asyncio
import aiohttp
import pandas as pd
from typing import List, Optional, Tuple

# File paths
# /home/huuquangdang/huu.quang.dang/thesis/deepfake/deepfake_backend/libs/tools/fake/fpp_real_v1.csv
input_csv = "/home/huuquangdang/huu.quang.dang/thesis/deepfake/deepfake_backend/libs/tools/real/fpp_real_v1.csv"
output_csv = "/home/huuquangdang/huu.quang.dang/thesis/deepfake/deepfake_backend/libs/tools/real/fpp_real_v2.csv"
folder_path = "/home/huuquangdang/huu.quang.dang/thesis/Dataset/facep/faceplus_processed_research_final_2_crop/real"

# API endpoint
analyze_api = "http://127.0.0.1:8000/api/detection/analyze-frame"

# Settings
TIMEOUT = 60
MAX_RETRIES = 2


def quicksort(arr, key_func):
    """
    Quicksort implementation for sorting CSV rows.
    
    Args:
        arr: List of rows to sort
        key_func: Function to extract the sort key from each row
    
    Returns:
        Sorted list of rows
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    pivot_key = key_func(pivot)
    
    left = [row for row in arr if key_func(row) < pivot_key]
    middle = [row for row in arr if key_func(row) == pivot_key]
    right = [row for row in arr if key_func(row) > pivot_key]
    
    return quicksort(left, key_func) + middle + quicksort(right, key_func)


def extract_filename_key(row):
    """
    Extract the filename from a row for sorting.
    Handles both successful rows and failed rows (filename,False).
    """
    if row:
        return row[0]  # First column is the filename
    return ""


def is_failed_row(row):
    """
    Check if a row is a failed entry (filename,False or filename,false).
    """
    return len(row) == 2 and (row[1] == 'False' or row[1] == 'false' or row[1] == False)


def is_all_zeros_row(row):
    """
    Check if a row has consecutive zeros in the beginning data columns.
    This includes cases like: filename,1.0,0.0,0.0,0.0,0.0,...
    We check if columns 2-5 (indices 1-4) are all zeros.
    """
    if len(row) <= 5:
        return False
    
    try:
        # Check columns 2-5 (indices 1-4): should be checking after first column
        # Pattern: filename, col1, col2, col3, col4, col5, ...
        # If col2, col3, col4, col5 (indices 1-4) are all 0.0, it's problematic
        check_cols = row[1:5]  # Get columns at indices 1, 2, 3, 4
        
        zero_count = 0
        for val in check_cols:
            try:
                if float(val) == 0.0:
                    zero_count += 1
            except (ValueError, TypeError):
                # If we can't convert to float, it's not a zero
                pass
        
        # If we have 3 or more consecutive zeros in the first 4 data columns, it's problematic
        # This catches: 1.0,0.0,0.0,0.0,0.0 or 0.0,0.0,0.0,0.0
        return zero_count >= 3
    except:
        return False


def is_low_confidence_row(row):
    """
    Check if a row has low confidence score (less than 0.8).
    The confidence is in the 4th column (index 3) after filename.
    Row format: filename, frame, face_id, timestamp, confidence, ...
    So confidence is at index 4 (0-based).
    """
    if len(row) <= 4:
        return False
    
    try:
        # The 4th column (index 3) after filename is the confidence
        # Actually looking at the format: filename,1.0,0.0,0.0,0.98
        # It seems: filename, col1, col2, col3, confidence
        confidence = float(row[4])  # Index 4 is the 5th column (confidence)
        return confidence < 0.8
    except (ValueError, TypeError, IndexError):
        return False


def needs_retry(row):
    """
    Check if a row needs to be retried (failed, all zeros, or low confidence).
    """
    return is_failed_row(row) or is_all_zeros_row(row) or is_low_confidence_row(row)


async def process_image(session, file_path, filename, retry_count=0) -> Tuple[str, Optional[List]]:
    """Process a single image: analyze-frame and extract CSV data."""
    try:
        # Call analyze-frame API
        with open(file_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=filename, content_type='image/jpeg')
            
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            async with session.post(analyze_api, data=data, timeout=timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[ERROR] Analyze-frame failed for {filename}: {error_text}")
                    
                    if retry_count < MAX_RETRIES:
                        print(f"[RETRY] Retrying {filename} (attempt {retry_count + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(1)
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


async def retry_problematic_rows(rows):
    """
    Retry rows that are failed or have all zeros - synchronously (one by one).
    """
    problematic_indices = []
    problematic_filenames = []
    
    # Find problematic rows
    for idx, row in enumerate(rows):
        if needs_retry(row):
            problematic_indices.append(idx)
            problematic_filenames.append(row[0])
    
    if not problematic_filenames:
        print("No problematic rows found!")
        return rows
    
    print(f"\n{'='*60}")
    print(f"Found {len(problematic_filenames)} problematic rows to retry")
    print(f"Processing synchronously (one by one) to ensure success")
    print(f"{'='*60}\n")
    
    # Retry problematic images synchronously
    connector = aiohttp.TCPConnector(limit=1, limit_per_host=1)
    timeout = aiohttp.ClientTimeout(total=None)
    
    updated_count = 0
    still_failed_count = 0
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for idx, (row_idx, filename) in enumerate(zip(problematic_indices, problematic_filenames), 1):
            file_path = os.path.join(folder_path, filename)
            
            print(f"[{idx}/{len(problematic_filenames)}] Processing: {filename}")
            
            if not os.path.exists(file_path):
                print(f"[WARNING] File not found: {file_path}")
                still_failed_count += 1
                continue
            
            try:
                fname, data_row = await process_image(session, file_path, filename)
                
                if data_row:
                    rows[row_idx] = [fname] + data_row
                    print(f"[✓] {fname} - Successfully updated")
                    updated_count += 1
                else:
                    print(f"[✗] {fname} - Still failed")
                    still_failed_count += 1
            except Exception as e:
                print(f"[✗] {filename} - Exception during retry: {e}")
                still_failed_count += 1
            
            # Small delay between requests to avoid overwhelming the server
            await asyncio.sleep(0.2)
    
    print(f"\n[RETRY SUMMARY]")
    print(f"  Updated: {updated_count}")
    print(f"  Still failed: {still_failed_count}")
    
    return rows


def remove_duplicates(rows):
    """
    Remove duplicate entries, keeping successful ones over failed ones.
    If duplicate filename exists, keep the one with full data (not False).
    """
    filename_dict = {}
    
    for row in rows:
        if not row:
            continue
            
        filename = row[0]
        
        # If filename not seen yet, add it
        if filename not in filename_dict:
            filename_dict[filename] = row
        else:
            # If we already have this filename, keep the successful one
            existing_row = filename_dict[filename]
            
            # If existing is failed and current is successful, replace
            if is_failed_row(existing_row) and not is_failed_row(row):
                filename_dict[filename] = row
            # If both are successful, keep the first one (or you could keep the last one)
            # If current is failed and existing is successful, do nothing (keep existing)
    
    return list(filename_dict.values())


async def async_main():
    print(f"Reading CSV file: {input_csv}")
    
    # Read all rows from the CSV
    rows = []
    with open(input_csv, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
    
    total_rows = len(rows)
    print(f"Total rows read: {total_rows}")
    
    if total_rows == 0:
        print("No data to sort!")
        return
    
    # Remove duplicates (keeping successful entries)
    print("Removing duplicates (keeping successful entries)...")
    unique_rows = remove_duplicates(rows)
    removed_count = total_rows - len(unique_rows)
    print(f"Removed {removed_count} duplicate entries")
    
    # Count problematic entries before retry
    failed_count = sum(1 for row in unique_rows if is_failed_row(row))
    zeros_count = sum(1 for row in unique_rows if is_all_zeros_row(row))
    low_conf_count = sum(1 for row in unique_rows if is_low_confidence_row(row))
    print(f"Failed entries (False): {failed_count}")
    print(f"All-zeros entries: {zeros_count}")
    print(f"Low confidence entries (< 0.8): {low_conf_count}")
    
    # Retry problematic rows (failed, all zeros, or low confidence)
    if failed_count > 0 or zeros_count > 0 or low_conf_count > 0:
        unique_rows = await retry_problematic_rows(unique_rows)
    
    # Recount after retry
    failed_rows = [row for row in unique_rows if is_failed_row(row)]
    zeros_rows = [row for row in unique_rows if is_all_zeros_row(row)]
    low_conf_rows = [row for row in unique_rows if is_low_confidence_row(row)]
    success_rows = [row for row in unique_rows if not needs_retry(row)]
    print(f"\nAfter retry:")
    print(f"Successful entries: {len(success_rows)}")
    print(f"Failed entries (still): {len(failed_rows)}")
    print(f"All-zeros entries (still): {len(zeros_rows)}")
    print(f"Low confidence entries (still): {len(low_conf_rows)}")
    
    # Sort using quicksort
    print("\nSorting rows using quicksort...")
    sorted_rows = quicksort(unique_rows, extract_filename_key)
    
    # Write sorted rows to output file
    print(f"Writing sorted and deduplicated data to: {output_csv}")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sorted_rows)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total rows processed: {total_rows}")
    print(f"Duplicates removed: {removed_count}")
    print(f"Final row count: {len(sorted_rows)}")
    print(f"  - Successful: {len(success_rows)}")
    print(f"  - Failed: {len(failed_rows)}")
    print(f"  - All-zeros: {len(zeros_rows)}")
    print(f"  - Low confidence: {len(low_conf_rows)}")
    print(f"Output file: {output_csv}")
    print(f"{'='*60}")
    
    # Display first few entries
    print("\nFirst 5 entries:")
    for i, row in enumerate(sorted_rows[:5], 1):
        if is_failed_row(row):
            print(f"  {i}. {row[0]} - FAILED")
        elif is_all_zeros_row(row):
            print(f"  {i}. {row[0]} - ALL ZEROS")
        elif is_low_confidence_row(row):
            conf = row[4] if len(row) > 4 else "?"
            print(f"  {i}. {row[0]} - LOW CONFIDENCE ({conf})")
        elif len(row) > 4:
            conf = row[4]
            print(f"  {i}. {row[0]} - Confidence: {conf}")
        else:
            print(f"  {i}. {row[0]}")
    
    print("\nLast 5 entries:")
    for i, row in enumerate(sorted_rows[-5:], len(sorted_rows)-4):
        if is_failed_row(row):
            print(f"  {i}. {row[0]} - FAILED")
        elif is_all_zeros_row(row):
            print(f"  {i}. {row[0]} - ALL ZEROS")
        elif is_low_confidence_row(row):
            conf = row[4] if len(row) > 4 else "?"
            print(f"  {i}. {row[0]} - LOW CONFIDENCE ({conf})")
        elif len(row) > 4:
            conf = row[4]
            print(f"  {i}. {row[0]} - Confidence: {conf}")
        else:
            print(f"  {i}. {row[0]}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
