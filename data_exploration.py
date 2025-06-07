# Load and examine the training data structure
import json
import pandas as pd

# Load data using pandas as specified in README
train_df = pd.read_json('A2-NLP_244/train.json')
val_df = pd.read_json('A2-NLP_244/val.json')

with open('A2-NLP_244/file_contents.json', 'r') as f:
    file_contents = json.load(f)

# Explore the data structure
print("=== TRAINING DATA STRUCTURE ===")
print(f"Training data shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")
print(f"First few annotations:")
print(train_df.head(3))

print(f"\n=== VALIDATION DATA STRUCTURE ===")
print(f"Validation data shape: {val_df.shape}")
print(val_df.head(3))

print(f"\n=== FILE CONTENTS STRUCTURE ===")
print(f"Number of files: {len(file_contents)}")
print(f"File IDs: {list(file_contents.keys())[:3]}")

# Look at annotation tags
print(f"\n=== ANNOTATION TAGS ===")
tag_counts = train_df['tag'].value_counts()
print(tag_counts)

# Examine a specific file and its annotations
print(f"\n=== EXAMPLE FILE + ANNOTATIONS ===")
sample_fileid = train_df.iloc[0]['fileid']
sample_text = file_contents[sample_fileid]
print(f"File ID: {sample_fileid}")
print(f"Text length: {len(sample_text)} characters")
print(f"First 300 chars:\n{sample_text[:300]}...")

# Find annotations for this file
file_annotations = train_df[train_df['fileid'] == sample_fileid]
print(f"\nAnnotations for this file: {len(file_annotations)}")
for idx, row in file_annotations.head(5).iterrows():
    start, end = row['start'], row['end']
    text_span = sample_text[start:end]
    print(f"  {row['tag']}: '{text_span}' (chars {start}-{end})")