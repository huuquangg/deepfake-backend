import os
import csv
import asyncio
import aiohttp

# ---------------- CONFIG ----------------
FOLDER_PATH = "/home/huuquangdang/huu.quang.dang/thesis/Dataset/celeb_df_crop/real"
OUTPUT_CSV = "merged_vectors_real.csv"

ANALYZE_API = "http://127.0.0.1:8000/api/detection/analyze-frame"
MERGE_API = "http://127.0.0.1:8000/api/detection/merge-openface-with-mobilenet?model_name=mobilenet"
# ----------------------------------------

async def process_image(session, file_path, filename):
    """Process a single image: analyze-frame -> merge vector."""
    try:
        # Step 1: analyze-frame
        with open(file_path, "rb") as f:
            data = {"file": f}
            async with session.post(ANALYZE_API, data=data) as resp:
                if resp.status != 200:
                    print(f"[ERROR] Analyze-frame failed for {filename}: {await resp.text()}")
                    return filename, None
                await resp.json()  # just wait for completion
                print(f"[INFO] Analyze-frame done: {filename}")

        # Step 2: merge-openface-with-mobilenet
        with open(file_path, "rb") as f:
            data = {"file": f}
            async with session.post(MERGE_API, data=data) as resp:
                if resp.status != 200:
                    print(f"[ERROR] Merge failed for {filename}: {await resp.text()}")
                    return filename, None
                merge_result = await resp.json()
                merged_vector = merge_result.get("merged_vector", None)
                return filename, merged_vector

    except Exception as e:
        print(f"[EXCEPTION] {filename}: {e}")
        return filename, None


async def main():
    # Get sorted list of all JPG images
    files_list = sorted(f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".jpg"))
    total_files = len(files_list)
    print(f"Found {total_files} images to process in sequence.\n")

    # Open CSV for writing
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        async with aiohttp.ClientSession() as session:
            for idx, filename in enumerate(files_list, start=1):
                file_path = os.path.join(FOLDER_PATH, filename)
                progress = idx / total_files * 100
                print(f"[{idx}/{total_files}] Processing: {filename} ({progress:.2f}%)")

                fname, merged_vector = await process_image(session, file_path, filename)

                if merged_vector:
                    writer.writerow([fname] + merged_vector)
                    print(f"[INFO] Merged vector appended for {fname}\n")
                else:
                    print(f"[WARN] Skipped {fname} due to API error\n")

    print(f"All done! Merged vectors saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
