import subprocess
import os
from datetime import datetime
import pandas as pd

# Configuration
KEYWORDS = [
    "​rupiah melemah",
    "​rupiah anjlok",
    "​rupiah terdepresiasi",
    "​nilai tukar rupiah terkini",
    "​kurs usd idr hari ini"
]
SINCE = "2026-01-01"
UNTIL = datetime.now().strftime('%Y-%m-%d')
LIMIT = 5000
# Gunakan path absolut dari directory kerja saat ini
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "tweets-data")
AUTH_TOKEN = os.getenv("TWITTER_AUTHTOKEN")

def run_scraper():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not AUTH_TOKEN:
        raise ValueError("TWITTER_AUTHTOKEN environment variable is not set")

    for keyword in KEYWORDS:
        safe_filename = keyword.lower().replace(" ", "_") + ".csv"
        # Pastikan kita memberikan path file yang benar ke tweet-harvest
        filepath = os.path.join(OUTPUT_DIR, safe_filename)
        search_query = f"{keyword} since:{SINCE} until:{UNTIL}"
        
        print(f"Scraping: {search_query}")
        
        # Perintah ke tweet-harvest
        cmd = [
            "npx", "--yes", "tweet-harvest",
            "-o", safe_filename, # tweet-harvest akan menyimpan di working dir, kita set working dir ke output
            "-s", search_query,
            "-l", str(LIMIT),
            "--token", AUTH_TOKEN
        ]
        
        try:
            # Jalankan di dalam folder output agar file tersimpan disana
            subprocess.run(cmd, check=True, cwd=OUTPUT_DIR)
            print(f"Saved to {filepath}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to scrape {keyword}: {e}")

def process_data():
    print(f"DEBUG: Checking for CSV files in {OUTPUT_DIR}")
    all_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
    print(f"DEBUG: Found files: {all_files}")
    dataframes = []
    
    for file in all_files:
        try:
            df = pd.read_csv(file)
            print(f"DEBUG: Read {file}, shape: {df.shape}")
            if not df.empty:
                dataframes.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    if dataframes:
        combined_df = pd.concat(dataframes).drop_duplicates().reset_index(drop=True)
        output_excel = os.path.join(BASE_DIR, "BoP_Indonesia_Scraped_Data.xlsx")
        combined_df.to_excel(output_excel, index=False)
        print(f"Data processed and saved to {output_excel}")
    else:
        print("No valid data to process from CSV files.")

if __name__ == "__main__":
    run_scraper()
    process_data()
