import os

total_tokens = 0
vocab = set()
file_count = 0

# List of your specific files to ensure we only count the clean data
files = [
    "cleaned_academic_regulations_text.txt",
    "cleaned_btech_page.txt",
    "cleaned_courses_text.txt",
    "cleaned_crf_page.txt",
    "cleaned_cse_department_page.txt",
    "cleaned_faculty_page.txt",
    "cleaned_phc_iitj.txt"
]

for filename in files:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            words = f.read().split()
            total_tokens += len(words)
            vocab.update(words)
            file_count += 1

print(f"--- Dataset Statistics ---")
print(f"Total Documents: {file_count}")
print(f"Total Tokens: {total_tokens}")
print(f"Vocabulary Size: {len(vocab)}")