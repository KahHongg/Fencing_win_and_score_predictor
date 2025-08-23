from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv

# List of competitions
competitions = [
    {"season": "2023/2024", "type": "World Cup", "url": "https://fie.org/competitions/2024/474"},
    {"season": "2023/2024", "type": "World Cup", "url": "https://fie.org/competitions/2024/160"},
    {"season": "2023/2024", "type": "World Cup", "url": "https://fie.org/competitions/2024/156"},
    {"season": "2023/2024", "type": "World Cup", "url": "https://fie.org/competitions/2024/163"},
    {"season": "2023/2024", "type": "World Cup", "url": "https://fie.org/competitions/2024/1410"},
    {"season": "2023/2024", "type": "Grand Prix", "url": "https://fie.org/competitions/2024/165"},
    {"season": "2023/2024", "type": "Grand Prix", "url": "https://fie.org/competitions/2024/1432"},
    {"season": "2023/2024", "type": "Grand Prix", "url": "https://fie.org/competitions/2024/158"},
    {"season": "2022/2023", "type": "World Championships", "url": "https://fie.org/competitions/2023/246"},
    {"season": "2022/2023", "type": "Grand Prix", "url": "https://fie.org/competitions/2023/165"},
    {"season": "2022/2023", "type": "Grand Prix", "url": "https://fie.org/competitions/2023/1432"},
    {"season": "2022/2023", "type": "Grand Prix", "url": "https://fie.org/competitions/2023/158"},
    {"season": "2022/2023", "type": "World Cup", "url": "https://fie.org/competitions/2023/474"},
    {"season": "2022/2023", "type": "World Cup", "url": "https://fie.org/competitions/2023/160"},
    {"season": "2022/2023", "type": "World Cup", "url": "https://fie.org/competitions/2023/156"},
    {"season": "2022/2023", "type": "World Cup", "url": "https://fie.org/competitions/2023/163"},
    {"season": "2022/2023", "type": "World Cup", "url": "https://fie.org/competitions/2023/1410"},
    {"season": "2021/2022", "type": "World Cup", "url": "https://fie.org/competitions/2022/474"},
    {"season": "2021/2022", "type": "World Cup", "url": "https://fie.org/competitions/2022/160"},
    {"season": "2021/2022", "type": "World Cup", "url": "https://fie.org/competitions/2022/164"},
    {"season": "2021/2022", "type": "Grand Prix", "url": "https://fie.org/competitions/2022/162"},
    {"season": "2021/2022", "type": "Grand Prix", "url": "https://fie.org/competitions/2022/158"},
    {"season": "2021/2022", "type": "World Championship", "url": "https://fie.org/competitions/2022/246"},
]

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Initialize the WebDriver
driver = webdriver.Chrome(options=chrome_options)

def get_competition_filename(comp):
    """Create filename from competition type and URL ID."""
    comp_type = comp['type'].replace(' ', '_')
    comp_id = comp['url'].split('/')[-1]
    season = comp['season'].replace('/', '_')
    return f"{comp_type}_{season}_{comp_id}"

try:
    for comp in competitions:
        url = comp['url']
        season = comp['season']
        comp_type = comp['type']
        print(f"\nProcessing {comp_type} ({season}): {url}")
        
        # Get filename for CSVs
        filename_base = get_competition_filename(comp)
        print(f"Filename base: {filename_base}")
        
        # Navigate to Athletes tab
        driver.get(url)
        time.sleep(5)
        
        try:
            # Extract Athletes data
            athletes_tab = driver.find_element(By.XPATH, "//div[contains(@class, 'Tabs-nav-link') and .//span[text()='Athletes']]")
            athletes_tab.click()
            print("Clicked on Athletes tab")
            time.sleep(3)
            
            table = driver.find_element(By.XPATH, "//table[contains(@class, 'table')]")
            headers = table.find_elements(By.TAG_NAME, "th")
            header_texts = [header.text.strip() for header in headers]
            
            rows = table.find_elements(By.XPATH, ".//tbody/tr")[:16]
            athletes_data = []
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                row_data = []
                for cell in cells:
                    if cell.find_elements(By.CLASS_NAME, "ResultsPool-flag"):
                        try:
                            nationality = cell.find_element(By.XPATH, ".//div/span[contains(@class, 'Flag-icon')]").get_attribute("class")
                            country_code = nationality.split("--")[-1].upper()
                            row_data.append(country_code)
                        except:
                            row_data.append("")
                    else:
                        row_data.append(cell.text.strip())
                athletes_data.append(row_data)
            
            athletes_csv = f"{filename_base}_athletes.csv"
            with open(athletes_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header_texts)
                writer.writerows(athletes_data)
            print(f"Saved top 16 athletes to {athletes_csv}")
            
            # Navigate to Results -> Pools Results tab
            driver.get(url)
            time.sleep(5)
            results_tab = driver.find_element(By.XPATH, "//div[contains(@class, 'Tabs-nav-link') and .//span[text()='Results']]")
            results_tab.click()
            print("Clicked on Results tab")
            time.sleep(3)
            
            pools_results_tab = driver.find_element(By.XPATH, "//div[contains(@class, 'Subtabs-nav-link') and .//span[text()='Pools Results']]")
            pools_results_tab.click()
            print("Clicked on Pools Results tab")
            time.sleep(3)
            
            # Extract pools results table
            pools_table = driver.find_element(By.XPATH, "/html/body/div[4]/div/div/div[2]/div[3]/div/div/div[3]/div[2]/div//table[contains(@class, 'table')]")
            pools_headers = pools_table.find_elements(By.TAG_NAME, "th")
            pools_header_texts = [header.text.strip().replace('\n', ' ') for header in pools_headers if header.text.strip() != 'Nation']  # Exclude Nation header
            
            pools_rows = pools_table.find_elements(By.XPATH, ".//tbody/tr")
            pools_data = []
            for row in pools_rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                row_data = []
                for i, cell in enumerate(cells):
                    if i == 3:  # Skip Nation column (index 3 based on provided HTML)
                        continue
                    if i == 1:  # Name column
                        name = cell.text.split('\n')[0].strip() if '\n' in cell.text else cell.text.strip()
                        row_data.append(name)
                    elif cell.find_elements(By.CLASS_NAME, "fa-arrow-up"):
                        row_data.append("Qualified")
                    else:
                        row_data.append(cell.text.strip())
                pools_data.append(row_data)
            
            pools_csv = f"{filename_base}_pools_results.csv"
            with open(pools_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(pools_header_texts)
                writer.writerows(pools_data)
            print(f"Saved pools results to {pools_csv}")
            
            # Navigate to Final Ranking tab
            final_ranking_tab = driver.find_element(By.XPATH, "//div[contains(@class, 'Subtabs-nav-link') and .//span[text()='Final Ranking']]")
            final_ranking_tab.click()
            print("Clicked on Final Ranking tab")
            time.sleep(3)
            
            # Extract final ranking table
            final_table = driver.find_element(By.XPATH, "/html/body/div[4]/div/div/div[2]/div[3]/div/div/div[7]/div[2]/div/div/div//table[contains(@class, 'table')]")
            final_headers = final_table.find_elements(By.TAG_NAME, "th")
            final_header_texts = [header.text.strip().replace('\n', ' ') for header in final_headers]
            
            final_rows = final_table.find_elements(By.XPATH, ".//tbody/tr")
            final_data = []
            for row in final_rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                row_data = []
                for cell in cells:
                    if cell.find_elements(By.CLASS_NAME, "ResultsPool-flag"):
                        try:
                            nationality = cell.find_element(By.XPATH, ".//div/span[contains(@class, 'Flag-icon')]").get_attribute("class")
                            country_code = nationality.split("--")[-1].upper()
                            row_data.append(country_code)
                        except:
                            row_data.append("")
                    else:
                        text = cell.text.strip()
                        if '\n' in text and cell.get_attribute("class") == "pools-results-table--cell-md":
                            name = text.split('\n')[0].strip()
                            row_data.append(name)
                        else:
                            row_data.append(text)
                final_data.append(row_data)
            
            final_csv = f"{filename_base}_final_ranking.csv"
            with open(final_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(final_header_texts)
                writer.writerows(final_data)
            print(f"Saved final ranking to {final_csv}")
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue
            
finally:
    driver.quit()
    print("\nBrowser closed")