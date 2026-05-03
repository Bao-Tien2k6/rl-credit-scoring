import os
import pandas as pd
import kagglehub
import shutil

# 1. check current directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Check if we're already inside a 'data' folder
if os.path.basename(current_dir) == "data":
    save_dir = current_dir
else:
    save_dir = os.path.join(current_dir, "data", "raw_data")
    os.makedirs(save_dir, exist_ok=True)

print(f"Data will be saved to: {save_dir}")

# 2. download dataset
print("Downloading LendingClub data from Kaggle...")
path = kagglehub.dataset_download("wordsforthewise/lending-club")
print(f"Downloaded to: {path}")

# 3. find csv file
accepted_source = os.path.join(
    path,
    "accepted_2007_to_2018q4",
    "accepted_2007_to_2018Q4.csv"
)

rejected_source = os.path.join(
    path,
    "rejected_2007_to_2018q4",
    "rejected_2007_to_2018Q4.csv"
)

if not os.path.exists(rejected_source):
    raise FileNotFoundError(f"CSV not found at {rejected_source}")

if not os.path.exists(accepted_source):
    raise FileNotFoundError(f"CSV not found at {accepted_source}")

# 4. copy file in current directory
accepted_dest = os.path.join(save_dir, "accepted_2007_to_2018Q4.csv")
rejected_dest = os.path.join(save_dir, "rejected_2007_to_2018Q4.csv")

if not os.path.exists(accepted_dest):
    print("Copying file to data directory...")
    shutil.copy(accepted_source, accepted_dest)
else:
    print("Accepted csv already exists in data directory.")

if not os.path.exists(rejected_dest):
    print("Copying file to data directory...")
    shutil.copy(rejected_source, rejected_dest)
else:
    print("rejected csv already exists in data directory.")