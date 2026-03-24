import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import preprocess, extract_skills

# ── Load Dataset ──────────────────────────────────────────
df = pd.read_csv('data/resumes.csv')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Total Resumes  : {len(df)}")
print(f"Columns        : {list(df.columns)}")
print(f"\nJob Categories : {df['Category'].nunique()}")
print(f"\nCategories:\n{df['Category'].value_counts()}")

# ── Preview Raw Resume ────────────────────────────────────
print("\n" + "=" * 50)
print("SAMPLE RAW RESUME (first 300 chars)")
print("=" * 50)
print(df['Resume_str'][0][:300])

# ── Preprocess One Resume ─────────────────────────────────
print("\n" + "=" * 50)
print("AFTER PREPROCESSING")
print("=" * 50)
processed = preprocess(df['Resume_str'][0])
print(processed[:300])

# ── Extract Skills ────────────────────────────────────────
print("\n" + "=" * 50)
print("SKILLS EXTRACTED")
print("=" * 50)
skills = extract_skills(df['Resume_str'][0])
print(skills)

# ── Preprocess Full Dataset ───────────────────────────────
print("\n" + "=" * 50)
print("PREPROCESSING FULL DATASET...")
print("=" * 50)
df['cleaned_resume'] = df['Resume_str'].apply(preprocess)
df['skills'] = df['Resume_str'].apply(extract_skills)

# Save cleaned dataset
df.to_csv('data/cleaned_resumes.csv', index=False)
print("✅ Cleaned dataset saved to data/cleaned_resumes.csv")
print(f"\nSample cleaned resume:\n{df['cleaned_resume'][0][:300]}")