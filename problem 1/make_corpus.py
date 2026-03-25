import os

# Define the directory and the output filename
data_folder = "./iitj_clean_data"
output_file = "corpus.txt"

# The specific list of cleaned files to combine
files = [
    "cleaned_academic_regulations_text.txt",
    "cleaned_btech_page.txt",
    "cleaned_courses_text.txt",
    "cleaned_crf_page.txt",
    "cleaned_cse_department_page.txt",
    "cleaned_faculty_page.txt",
    "cleaned_phc_iitj.txt"
]

def merge_files():
    merged_count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in files:
            filepath = os.path.join(data_folder, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as infile:
                    # Read content and add a newline to keep documents distinct
                    outfile.write(infile.read())
                    outfile.write("\n")
                print(f"Successfully added: {filename}")
                merged_count += 1
            else:
                print(f"File not found: {filename}")
    
    print(f"\nComplete! '{output_file}' created using {merged_count} files.")

if __name__ == "__main__":
    merge_files()