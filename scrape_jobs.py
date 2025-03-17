import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

def parse_release_date(text):
    match = re.search(r'(\d+)\s*(month|week|day|hour|minute|year)s?', text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return text

def parse_applicant_number(text):
    match = re.search(r'(\d+)', text)
    if match:
        return match.group(1)
    return "0"

def extract_details(details):
    """
    Verilen detaylar listesine (detail_texts) göre:
    - İlk öğe lokasyon,
    - İçerisinde "hour", "day", "week", "month", "year" geçen öğe yayınlanma süresi,
    - "clicked apply" veya "applicants" içeren öğe başvuru sayısı olarak ayıklar.
    """
    location = details[0] if len(details) > 0 else "No Location"
    release_date = "No Release Date"
    applicant_number = "No Applicant Number"
    
    for item in details:
        if re.search(r"(hour|day|week|month|year)s?\s*ago", item, re.IGNORECASE):
            release_date = item
        if re.search(r"(clicked apply|applicants)", item, re.IGNORECASE):
            applicant_number = item
    return location, release_date, applicant_number

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# 1) LinkedIn'e giriş
driver.get('https://www.linkedin.com/login')
time.sleep(2)
print("On login page...")

username_input = driver.find_element(By.ID, "username")
password_input = driver.find_element(By.ID, "password")
fake_username = "busela99@gmail.com"
fake_password = "busela123"
username_input.send_keys(fake_username)
password_input.send_keys(fake_password)
username_input.submit()
time.sleep(5)
print("LinkedIn'e giriş yapıldı.")

# 2) İş ilanları sayfasına geçiş
url = "https://www.linkedin.com/jobs/search/?keywords=Data%20Analyst&location=Turkey"
driver.get(url)
time.sleep(5)
print("Türkiye iş analisti sayfasına girildi.")

wait = WebDriverWait(driver, 10)
job_details_list = []

# -- 3) Toplam sayfa sayısını bulma --
# "ul.artdeco-pagination__pages--number" altındaki li etiketlerini alıyoruz
pagination_elems = driver.find_elements(By.CSS_SELECTOR, "ul.artdeco-pagination__pages--number li")
page_numbers = []

for elem in pagination_elems:
    txt = elem.text.strip()
    # Eğer '...' yerine gerçek bir sayı ise ekle
    if txt.isdigit():
        page_numbers.append(int(txt))

# Eğer hiçbir sayfa bulunamazsa varsayılan olarak 1 sayfa
if page_numbers:
    last_page = max(page_numbers)  # en büyük sayfa numarası
else:
    last_page = 1

print(f"Toplam {last_page} sayfa bulundu.")

def scrape_current_page():
    """
    Mevcut sayfadaki ilanları tıklayıp verileri çekerek job_details_list'e ekler.
    """
    # Sayfadaki ilan kartlarını bulalım
    job_cards = driver.find_elements(By.XPATH, "//li[contains(@class, 'scaffold-layout__list-item')]")
    print("Bu sayfada bulunan ilan kartı sayısı:", len(job_cards))
    
    for i, card in enumerate(job_cards):
        try:
            # Kart ekran içinde görünmüyorsa görünür konuma getiriyoruz
            driver.execute_script("arguments[0].scrollIntoView();", card)
            time.sleep(1)
            
            # İlgili job kartını tıklıyoruz
            card.click()
            
            # Hem iş ilanı başlığı hem de detay container'ın yüklenmesini bekliyoruz
            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "h1.t-24.t-bold.inline")))
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.job-details-jobs-unified-top-card__primary-description-container")))
            time.sleep(1)
            
            # Sayfa kaynak kodunu alıp BeautifulSoup ile parse ediyoruz
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            
            # İş ilanı başlığını çekiyoruz
            job_title_el = soup.find("h1", class_="t-24 t-bold inline")
            job_title = job_title_el.get_text(strip=True) if job_title_el else "No Title"
            
            # Detay container'ı çekiyoruz
            container = soup.find("div", class_="job-details-jobs-unified-top-card__primary-description-container")
            if container:
                spans = container.find_all("span")
                detail_texts = [span.get_text(strip=True) for span in spans]
            else:
                detail_texts = []
            
            # Lokasyon, tarih, başvuru sayısı çıkarma
            location_text, release_date_text, applicant_text = extract_details(detail_texts)
            release_date_parsed = parse_release_date(release_date_text)
            applicant_number_parsed = parse_applicant_number(applicant_text)
            
            # "ui-label text-body-small" Bilgileri (örnek: Remote / Full-time vb.)
            labels = soup.find_all("span", class_="ui-label text-body-small")
            if len(labels) >= 2:
                workplace_type = labels[0].get_text(strip=True)
                employment_type = labels[1].get_text(strip=True)
            else:
                workplace_type = "No workplace info"
                employment_type = "No employment info"
            
            # job_details_list'e ekle
            job_details_list.append({
                "job_title": job_title,
                "location": location_text,
                "release_date": release_date_parsed,
                "applicant_number": applicant_number_parsed,
                "workplace_type": workplace_type,
                "employment_type": employment_type
            })
            
            print(f"İlan {i+1}: {job_title} -> {location_text}, {release_date_parsed}, {applicant_number_parsed}, {workplace_type}, {employment_type}")
        
        except Exception as e:
            print(f"İlan {i+1} işlenirken hata oluştu: {str(e)}")


# -- 4) Tüm sayfaları gezip ilanları çekelim --
for page in range(1, last_page + 1):
    print(f"\n=== {page}. sayfa işleniyor ===")
    
    # Eğer ilk sayfada değilsek, ilgili sayfa düğmesine tıklayalım
    if page != 1:
        try:
            # Aria-label üzerinden "Sayfa X" butonunu bulalım (Türkçe arayüzde)
            # İngilizce arayüzde "Page X" olabilir, ona göre değiştirmeniz gerekebilir.
            page_button = driver.find_element(By.XPATH, f"//button[@aria-label='Page {page}']")
            driver.execute_script("arguments[0].click();", page_button)
            time.sleep(2)
        except Exception as e:
            print(f"{page}. sayfaya tıklanırken hata oluştu: {e}")
            break
    
    # Mevcut sayfadaki ilanları çekelim
    scrape_current_page()

# Tüm sayfalar tamamlandı, sonuçları yazdıralım
driver.quit()

print("\nTüm sayfaların işlenmesi tamamlandı.")
print("Toplam iş ilanı detayları:", len(job_details_list))
for job in job_details_list:
    print(job)


