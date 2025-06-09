import os
import pandas as pd

file_path = "C:\\CODE\\princy\\ai\\derma-wise\\skin_data\\model_concerns.csv"
print("File exists:", os.path.exists(file_path))
try:
    df = pd.read_csv(file_path, encoding="utf-8")
    print(df.head())  # Show first few rows
except Exception as e:
    print(f"Encoding issue: {e}")

with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    print("First few lines:\n", lines[:5])



