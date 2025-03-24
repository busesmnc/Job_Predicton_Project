import sqlite3
import json

def create_connection(db_file="job_data.db"):
    conn = sqlite3.connect(db_file)
    return conn

def create_tables(conn):
    cursor = conn.cursor()
    
    # Job_Categories tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Job_Categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE
    )
    ''')
    
    # Locations tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location_info TEXT UNIQUE
    )
    ''')
    
    # Employment_Types tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Employment_Types (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type_name TEXT UNIQUE
    )
    ''')
    
    # Workplace_Types tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Workplace_Types (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type_name TEXT UNIQUE
    )
    ''')
    
    # Job_Postings tablosu (normalize edilmiş değerler için foreign key'ler eklenmiştir)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Job_Postings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_id INTEGER,
        job_title TEXT,
        location_id INTEGER,
        release_date TEXT,
        applicant_number TEXT,
        workplace_type_id INTEGER,
        employment_type_id INTEGER,
        job_description TEXT,
        FOREIGN KEY (category_id) REFERENCES Job_Categories(id),
        FOREIGN KEY (location_id) REFERENCES Locations(id),
        FOREIGN KEY (workplace_type_id) REFERENCES Workplace_Types(id),
        FOREIGN KEY (employment_type_id) REFERENCES Employment_Types(id)
    )
    ''')
    
    conn.commit()

def load_jobs_from_json(file_path="job_data.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        jobs = json.load(file)
    return jobs

def get_or_create_id(cursor, table, column, value):
    """Verilen değeri belirtilen tabloda arar; yoksa ekleyip id'sini döndürür."""
    cursor.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))
        return cursor.lastrowid

def save_jobs_to_db(job_details_list):
    conn = create_connection()
    cursor = conn.cursor()
    
    for job in job_details_list:
        # JSON anahtarlarını aynen kullanıyoruz:
        job_title = job.get("job_title", "No Title")
        location = job.get("location", "No Location")
        release_date = job.get("release_date", "No Date")
        applicant_number = job.get("applicant_number", "0")
        workplace_type = job.get("workplace_type", "No workplace info")
        employment_type = job.get("employment_type", "No employment info")
        job_description = job.get("job_description", "No Description")
        category = job.get("category", "Data Analyst")
        
        # Normalize edilmiş tablolara ekleme:
        category_id = get_or_create_id(cursor, "Job_Categories", "category_name", category)
        location_id = get_or_create_id(cursor, "Locations", "location_info", location)
        employment_type_id = get_or_create_id(cursor, "Employment_Types", "type_name", employment_type)
        workplace_type_id = get_or_create_id(cursor, "Workplace_Types", "type_name", workplace_type)
        
        # İş ilanını Job_Postings tablosuna ekleme
        cursor.execute('''
        INSERT INTO Job_Postings (
            category_id, job_title, location_id, release_date, applicant_number, workplace_type_id, employment_type_id, job_description
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (category_id, job_title, location_id, release_date, applicant_number, workplace_type_id, employment_type_id, job_description))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Adım 1: Veritabanı bağlantısı oluştur ve tabloları yarat
    conn = create_connection()
    create_tables(conn)
    conn.close()
    print("Tablolar başarıyla oluşturuldu.")
    
    # Adım 2: JSON dosyasından veriyi oku (dosya adı: job_data.json)
    job_details_list = load_jobs_from_json("job_data.json")
    
    # Adım 3: Verileri veritabanına kaydet
    save_jobs_to_db(job_details_list)
    print("Veriler veritabanına kaydedildi.")
