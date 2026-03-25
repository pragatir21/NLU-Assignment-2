import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
import re

# ==========================================
# Configuration and Setup
# ==========================================

# Define the directory where the scraped text files will be saved
OUTPUT_DIR = "./iitj_data"

# Create the output directory if it does not already exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Use a standard User-Agent header to prevent the server from blocking the request
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ==========================================
# Function 1: HTML Webpage Scraper
# ==========================================
def scrape_iitj_webpage(url, filename):
    """
    Scrapes paragraph text from a given HTML webpage and saves it to a .txt file.
    
    Args:
        url (str): The URL of the webpage to scrape.
        filename (str): The name of the output text file (e.g., 'academics.txt').
    """
    try:
        # Send an HTTP GET request to the specified URL
        response = requests.get(url, headers=HEADERS)
        
        # Check if the request was successful (Status Code 200)
        response.raise_for_status()
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from all paragraph (<p>) tags
        # We target <p> tags to avoid scraping navigation menus, footers, and raw HTML scripts
        paragraphs = soup.find_all('p')
        
        # Join the text from all paragraphs, separating them with a newline
        page_text = "\n".join([para.get_text() for para in paragraphs])
        
        # Clean up basic whitespace issues immediately (optional, as task 1 preprocessing handles more)
        page_text = re.sub(r'\s+', ' ', page_text).strip()
        
        # Define the full path for the output file
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # Write the extracted text to the file with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(page_text)
            
        print(f"Successfully scraped HTML: {url} -> {filename}")
        
    except requests.exceptions.RequestException as e:
        # Handle network-related errors (e.g., invalid URL, connection timeout)
        print(f"Failed to scrape {url}. Error: {e}")

# ==========================================
# Function 2: PDF Text Extractor
# ==========================================
def extract_text_from_pdf(pdf_path, filename):
    """
    Extracts text from a local PDF file (like Academic Regulations) and saves it to a .txt file.
    
    Args:
        pdf_path (str): The local file path to the downloaded PDF.
        filename (str): The name of the output text file (e.g., 'regulations.txt').
    """
    try:
        text_content = ""
        
        # Open the PDF file in binary read mode ('rb')
        with open(pdf_path, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Iterate through all the pages in the PDF
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                
                # Extract text from the current page and append it to our main string
                extracted_text = page.extract_text()
                if extracted_text:
                    text_content += extracted_text + "\n"
        
        # Define the full path for the output file
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # Write the extracted text to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_content)
            
        print(f"Successfully extracted PDF: {pdf_path} -> {filename}")
        
    except FileNotFoundError:
        print(f"Error: The PDF file '{pdf_path}' was not found. Please download it first.")
    except Exception as e:
        print(f"An error occurred while parsing the PDF: {e}")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    print("Starting data collection...")
    
    # 1. Scrape standard webpages 
    web_sources = [
        ("https://www.iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology", "btech_page.txt"),
        ("https://www.iitj.ac.in/computer-science-engineering/", "cse_department_page.txt"),
        ("https://www.iitj.ac.in/health-center/en/health-center", "phc_iitj.txt"),
        ("https://www.iitj.ac.in/People/List?dept=computer-science-engineering&c=1c8346b0-07a4-401e-bce0-7f571bb4fabd", "faculty_page.txt"),
        ("https://www.iitj.ac.in/crf/en/crf", "crf_page.txt")
    ]
    
    for url, out_name in web_sources:
        scrape_iitj_webpage(url, out_name)
        
    # 2. Process Local PDFs
    pdf_sources = [
        ("academic_regulations.pdf", "academic_regulations_text.txt"),
        ("CSE-Courses-Details.pdf", "courses_text.txt")
    ]
    
    for local_pdf_name, out_name in pdf_sources:
        if os.path.exists(local_pdf_name):
            extract_text_from_pdf(local_pdf_name, out_name)
        else:
            print(f"\nREMINDER: You MUST manually download the PDF '{local_pdf_name}'.")
            print(f"Save it as '{local_pdf_name}' in this folder to parse it.")
        
    print("\nData collection complete. You can now run the preprocessing script on the 'iitj_data' folder.")
