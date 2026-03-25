import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Define the folder where your cleaned files are stored
data_folder = "./iitj_clean_data"

files = [
    "cleaned_academic_regulations_text.txt",
    "cleaned_btech_page.txt",
    "cleaned_courses_text.txt",
    "cleaned_crf_page.txt",
    "cleaned_cse_department_page.txt",
    "cleaned_faculty_page.txt",
    "cleaned_phc_iitj.txt"
]

full_text = ""
for filename in files:
    # Create the full path (e.g., ./iitj_clean_data/cleaned_btech_page.txt)
    filepath = os.path.join(data_folder, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            full_text += f.read() + " "
    else:
        print(f"Warning: {filename} not found at {filepath}")

if full_text.strip():
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_text)
    
    # Save and display
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wordcloud.png')
    print("Word cloud saved as wordcloud.png!")
    plt.show()
else:
    print("Error: No text found. Ensure your files are located inside the 'iitj_clean_data' folder.")