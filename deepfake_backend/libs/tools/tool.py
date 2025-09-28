import os
import requests
import csv

# Folder containing your images
folder_path = "/home/huuquangdang/huu.quang.dang/thesis/Dataset/celeb_df_crop/fake"
output_csv = "merged_vectors.csv"

# API endpoints
analyze_api = "http://127.0.0.1:8000/api/detection/analyze-frame"
merge_api = "http://127.0.0.1:8000/api/detection/merge-openface-with-mobilenet?model_name=mobilenet"

# Get all JPG files and sort them alphanumerically
files_list = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])
total_files = len(files_list)
print(f"Found {total_files} images to process in sequence.")

# Prepare CSV file
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Loop through all images in folder
    for idx, filename in enumerate(files_list, start=1):
        file_path = os.path.join(folder_path, filename)
        print(f"[{idx}/{total_files}] Processing: {filename} ({idx/total_files*100:.2f}%)")

        # Step 1: Call analyze-frame API
        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "image/jpeg")}
                resp = requests.post(analyze_api, files=files)
                if resp.status_code != 200:
                    print(f"[ERROR] Analyze-frame failed for {filename}: {resp.text}")
                    continue
                _ = resp.json()  # just wait for completion
                print(f"[INFO] Analyze-frame done: {filename}")
        except Exception as e:
            print(f"[EXCEPTION] Analyze-frame error for {filename}: {e}")
            continue

        # Step 2: Call merge-openface-with-mobilenet API
        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "image/jpeg")}
                resp = requests.post(merge_api, files=files)
                if resp.status_code != 200:
                    print(f"[ERROR] Merge failed for {filename}: {resp.text}")
                    continue
                merge_result = resp.json()
        except Exception as e:
            print(f"[EXCEPTION] Merge API error for {filename}: {e}")
            continue

        # Step 3: Append merged vector to CSV
        merged_vector = merge_result.get("merged_vector", [])
        if merged_vector:
            writer.writerow([filename] + merged_vector)
            print(f"[INFO] Merged vector appended for {filename}")
        else:
            print(f"[WARN] No merged vector returned for {filename}")

print(f"All done! Merged vectors saved to {output_csv}")
