import json
import time
from langdetect import detect
from deep_translator import GoogleTranslator

def ensure_english(text):

    if not text or len(text.strip()) < 3 or not any(char.isalpha() for char in text):
        return text
    try:
        lang = detect(text)
        if lang != 'en':
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
        else:
            return text
    except Exception as e:
        print("Translation error:", e)
        return text

fields_to_translate = [
    "job_title",
    "location",
    "release_date",
    "applicant_number",
    "workplace_type",
    "employment_type",
    "job_description",
    "category"
]

with open('job_details.json', 'r', encoding='utf-8') as f:
    jobs = json.load(f)

# jobs = jobs[:10]

job_count = len(jobs)
print(f"Processing {job_count} jobs...")

for idx, job in enumerate(jobs, start=1):
    for field in fields_to_translate:
        if field in job and isinstance(job[field], str) and job[field].strip():
            job[field] = ensure_english(job[field])
            # API rate limitlerini aşmamak için kısa bekleme (0.5 saniye)
            time.sleep(0.5)
    print(f"Job {idx}/{job_count} processed.")
    print(job)
    print("-" * 50)

with open('job_data.json', 'w', encoding='utf-8') as f:
    json.dump(jobs, f, ensure_ascii=False, indent=4)

print("All fields translated, saved to job_data.json")
