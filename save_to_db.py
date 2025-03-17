import sqlite3
import json

def create_connection(db_file="job_data.db"):
    conn = sqlite3.connect(db_file)
    return conn

def create_tables(conn):
    cursor = conn.cursor()
    
    # Şirketler tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Companies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_name TEXT UNIQUE
    )
    ''')
    
    # İş Kategorileri tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Job_Categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE
    )
    ''')
    
    # İş İlanları tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Job_Postings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id INTEGER,
        category_id INTEGER,
        job_title TEXT,
        location TEXT,
        release_date TEXT,
        applicant_number TEXT,
        workplace_type TEXT,
        employment_type TEXT,
        FOREIGN KEY (company_id) REFERENCES Companies(id),
        FOREIGN KEY (category_id) REFERENCES Job_Categories(id)
    )
    ''')
    
    # Lokasyonlar tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location_info TEXT
    )
    ''')
    
    # Employment_Types tablosu (örn. Full-time, Part-time, Contract)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Employment_Types (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type_name TEXT UNIQUE
    )
    ''')
    
    # Workplace_Types tablosu (örn. Remote, On-site, Hybrid)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Workplace_Types (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type_name TEXT UNIQUE
    )
    ''')
    
    # Scraping Seansları tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Scraping_Sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_datetime TEXT,
        url TEXT,
        jobs_count INTEGER
    )
    ''')
    
    conn.commit()

def load_jobs_from_json(file_path="jobs.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        jobs = json.load(file)
    return jobs

def save_jobs_to_db(job_details_list):
    conn = create_connection()
    cursor = conn.cursor()
    
    for job in job_details_list:
        company_name = job.get("company_name", "No Company")
        job_title = job.get("job_title", "No Title")
        location = job.get("location", "No Location")
        release_date = job.get("release_date", "No Date")
        applicant_number = job.get("applicant_number", "0")
        workplace_type = job.get("workplace_type", "No workplace info")
        employment_type = job.get("employment_type", "No employment info")
        
        # Şirket bilgisini kontrol edip ekleme
        cursor.execute("SELECT id FROM Companies WHERE company_name = ?", (company_name,))
        result = cursor.fetchone()
        if result:
            company_id = result[0]
        else:
            cursor.execute("INSERT INTO Companies (company_name) VALUES (?)", (company_name,))
            company_id = cursor.lastrowid
        
        # İş kategorisini belirleme (örneğin sabit "Data Analyst" olarak)
        category_name = "Data Analyst"
        cursor.execute("SELECT id FROM Job_Categories WHERE category_name = ?", (category_name,))
        result = cursor.fetchone()
        if result:
            category_id = result[0]
        else:
            cursor.execute("INSERT INTO Job_Categories (category_name) VALUES (?)", (category_name,))
            category_id = cursor.lastrowid

        # İş ilanını ekleme
        cursor.execute('''
        INSERT INTO Job_Postings (
            company_id, category_id, job_title, location, release_date, applicant_number, workplace_type, employment_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (company_id, category_id, job_title, location, release_date, applicant_number, workplace_type, employment_type))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Adım 1: Veritabanı bağlantısı oluştur ve tabloları oluştur
    conn = create_connection()
    create_tables(conn)
    conn.close()
    print("Tablolar başarıyla oluşturuldu.")
    
    # Adım 2: JSON dosyasından veriyi oku
    job_details_list = load_jobs_from_json("jobs.json")
    
    # Adım 3: Verileri veritabanına kaydet
    save_jobs_to_db(job_details_list)
    print("Veriler veritabanına kaydedildi.")
