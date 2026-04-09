import time
import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
from webdriver_manager.chrome import ChromeDriverManager

# --- Configuration ---
DATABASE_FILE = 'threat_feed.db'
KEYWORDS = [
    'phishing', 'malware', 'scam', 'fake login', 'bank alert',
    'hacked', 'data breach', 'vulnerability', 'exploit', 'cyberattack',
    'ransomware', 'trojan', 'security flaw', 'zero-day'
]

# --- Database Setup ---
def setup_database():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            threat_score INTEGER NOT NULL,
            threat_level TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# --- Data Fetching ---
def fetch_data_from_sources():
    print("Fetching data from BleepingComputer using Selenium Stealth...")
    posts = []

    chrome_options = webdriver.ChromeOptions()
    # Run with GUI first (remove headless to avoid detection)
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    # Apply stealth to avoid detection
    stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )

    try:
        url = "https://www.bleepingcomputer.com/"
        driver.get(url)

        # Wait for main news container
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "bc-home-news-main-wrap"))
        )

        headlines = driver.find_elements(
            By.CSS_SELECTOR,
            "ul#bc-home-news-main-wrap li div.bc_latest_news_text h4 a"
        )

        print(f"Found {len(headlines)} headlines")
        if not headlines:
            with open("debug_page.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("No headlines found. Saved page to debug_page.html")

        for headline in headlines:
            text = headline.text.strip()
            print("-", text)
            posts.append({"source": "BleepingComputer", "content": text})

    except Exception as e:
        print(f"Error scraping blog with Selenium: {e}")
    finally:
        driver.quit()

    return posts

# --- Risk Scoring ---
def calculate_threat_score(post):
    score = 0
    content_lower = post['content'].lower()
    keyword_hits = sum([1 for keyword in KEYWORDS if keyword in content_lower])
    score += keyword_hits * 25
    if post['source'] == 'BleepingComputer':
        score += 20
    return min(score, 100)

# --- Monitoring Loop ---
def monitor_threats():
    setup_database()
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    print("--- AI Early Warning System Monitor Started ---")
    while True:
        posts = fetch_data_from_sources()
        if not posts:
            print("No headlines found in this cycle.")

        for post in posts:
            score = calculate_threat_score(post)
            level = "Low"
            if score >= 80:
                level = "High"
            elif score >= 50:
                level = "Medium"

            if score > 30:
                print(f"Threat Detected! Level: {level}, Score: {score}, Content: {post['content']}")
                cursor.execute(
                    "INSERT INTO alerts (source, content, threat_score, threat_level) VALUES (?, ?, ?, ?)",
                    (post['source'], post['content'], score, level)
                )
                conn.commit()

        print("Cycle complete. Waiting for 60 seconds...")
        time.sleep(60)

if __name__ == '__main__':
    monitor_threats()
