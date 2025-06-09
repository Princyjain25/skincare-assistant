import pandas as pd

# Load the CSV file
file_path = "data/SideHustleIdeasDataCSV.csv"
df = pd.read_csv(file_path)

# Check if the file is empty
if df.empty:
    raise ValueError("The CSV file is empty. Please check the file path and content.")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Rename incorrect column
df = df.rename(columns={"catergory": "category"})

# Optionally strip whitespace from column names
df.columns = df.columns.str.strip()

# Define expected columns
required_cols = [
    "side_hustle_idea", "description", "category",
    "initial_investment", "required_skills_or_qualifications"
]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# Handle missing values (fill with empty string or drop rows if necessary)
df.fillna("", inplace=True)

# Trim whitespace in text fields
text_columns = ["side_hustle_idea", "description", "category", "required_skills_or_qualifications"]
df[text_columns] = df[text_columns].apply(lambda x: x.str.strip())

# Standardize initial investment format
df["initial_investment"] = df["initial_investment"].str.replace("Low investment ", "Low ($0-$500)")
df["initial_investment"] = df["initial_investment"].str.replace("Medium investment ", "Medium ($500-$2000)")
df["initial_investment"] = df["initial_investment"].str.replace("High investment ", "High ($2000+)")

# Combine fields into a single searchable text
df["combined_text"] = (
    "Idea: " + df["side_hustle_idea"].str.strip() +
    ". Description: " + df["description"].str.strip() +
    ". Category: " + df["category"].str.strip() +
    ". Investment: " + df["initial_investment"].str.strip() +
    ". Skills: " + df["required_skills_or_qualifications"].str.strip()
)

# Drop rows with empty combined_text just in case
df = df[df["combined_text"].str.len() > 0].reset_index(drop=True)

# Print first few rows to verify cleaning
# print(df.head())

# Save cleaned data
cleaned_file_path = "data/cleaned_side_hustle_data.csv"
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")
