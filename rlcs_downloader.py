import os
import requests
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "cTWL9ikjuQ7Pyu2fIfnDwtM9n7PYZLcG8YtjTBzp"

# --- НАЛАШТУВАННЯ ---
TARGET_DOWNLOADS = 10
STATE_FILE = "last_page_state.txt"
MAX_WORKERS = 2  # Кількість одночасних завантажень (2-3 оптимально для free API)

os.makedirs("rlcs_replays", exist_ok=True)

# 1. ОПТИМІЗАЦІЯ: Використовуємо Session для повторного використання TCP-з'єднань
session = requests.Session()
session.headers.update({'Authorization': API_KEY})

# Блокувальник для безпечного оновлення лічильника між потоками
counter_lock = threading.Lock()
downloaded_count = 0

def sanitize_filename(name):
    """Очищає назву від заборонених символів OS."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

def download_replay(replay_id, replay_title, file_path):
    """Функція для завантаження одного файлу в окремому потоці."""
    global downloaded_count
    dl_url = f"https://ballchasing.com/api/replays/{replay_id}/file"
    
    retry = True
    while retry:
        dl_res = session.get(dl_url)

        if dl_res.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(dl_res.content)
            
            with counter_lock:
                downloaded_count += 1
                current_count = downloaded_count
            
            print(f"[{current_count}/{TARGET_DOWNLOADS}] 📥 Успішно: {replay_title}")
            retry = False
            time.sleep(1) # Невелика пауза між потоками
            
        elif dl_res.status_code == 429:
            print(f"⚠️ [429] Сервер гальмує потік для {replay_id}. Пауза 10с...")
            time.sleep(10)
        else:
            print(f"❌ Помилка {dl_res.status_code} для {replay_id}")
            retry = False

# --- ГОЛОВНИЙ ЦИКЛ ---
if __name__ == "__main__":
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            url = f.read().strip()
        print("🧠 Продовжуємо з останньої сторінки...")
    else:
        # count=200 (максимум API), жорстке сортування по даті матчу
        url = "https://ballchasing.com/api/replays?title=RLCS&pro=true&min-rank=grand-champion-2&count=200&sort-by=replay-date&sort-dir=desc"
        print("🔍 Починаємо масовий пошук з першої сторінки (по 200 матчів на сторінку)...")

    while url and downloaded_count < TARGET_DOWNLOADS:
        print(f"\n📄 Отримую список матчів...")
        response = session.get(url)

        if response.status_code == 200:
            data = response.json()
            replays = data.get('list', [])
            
            if not replays:
                print("Порожня сторінка.")
                break

            print(f"✅ На сторінці {len(replays)} матчів. Фільтрую...")
            
            # Збираємо список того, що реально треба качати
            tasks = []
            for replay in replays:
                if downloaded_count + len(tasks) >= TARGET_DOWNLOADS:
                    break

                replay_id = replay['id']
                raw_title = replay.get('title') or f"Match_{replay_id}"
                replay_title = sanitize_filename(raw_title)
                file_path = f"rlcs_replays/{replay_title}_{replay_id}.replay"

                if os.path.exists(file_path):
                    print(f"⏩ Вже є: {replay_title}")
                else:
                    tasks.append((replay_id, replay_title, file_path))

            # 2. ОПТИМІЗАЦІЯ: Багатопотокове завантаження
            if tasks:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(download_replay, rid, title, path) for rid, title, path in tasks]
                    # Чекаємо завершення всіх потоків поточної сторінки
                    for future in as_completed(futures):
                        pass

            # Збереження прогресу сторінки
            if downloaded_count < TARGET_DOWNLOADS:
                next_url = data.get('next')
                if next_url:
                    url = next_url
                    with open(STATE_FILE, "w") as f:
                        f.write(url)
                    print(f"💾 Прогрес збережено. Перехід на наступну сторінку.")
                else:
                    print("🏁 Це остання сторінка.")
                    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
                    break

        elif response.status_code == 429:
            print("⚠️ [429] Ліміт на список. Пауза 15с...")
            time.sleep(15)
        else:
            print(f"❌ Помилка API: {response.status_code}")
            break

    print(f"\n🎉 Завершено! Завантажено {downloaded_count} нових реплеїв.")