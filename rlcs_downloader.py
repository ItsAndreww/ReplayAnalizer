import os
import requests
import time

API_KEY = "cTWL9ikjuQ7Pyu2fIfnDwtM9n7PYZLcG8YtjTBzp"

# --- НАЛАШТУВАННЯ ---
TARGET_DOWNLOADS = 100  # Скільки НОВИХ реплеїв ти хочеш скачати за цей запуск (можеш змінити)
STATE_FILE = "last_page_state.txt"  # Файл, де скрипт зберігатиме свою пам'ять

os.makedirs("rlcs_replays", exist_ok=True)
headers = {'Authorization': API_KEY}

# 1. Відновлюємо пам'ять (якщо є збережений прогрес)
if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as f:
        url = f.read().strip()
    print("🧠 Знайдено збережений прогрес! Продовжуємо з останньої сторінки...")
else:
    url = "https://ballchasing.com/api/replays?title=rlcs&pro=true&min-rank=supersonic-legend&count=50"
    print("🔍 Починаємо пошук RLCS реплеїв з найпершої сторінки...")

downloaded_count = 0

# 2. Головний цикл гортання сторінок
while url and downloaded_count < TARGET_DOWNLOADS:
    print(f"\n📄 Завантажую список матчів зі сторінки...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        replays = data.get('list', [])
        
        if not replays:
            print("Нових реплеїв на цій сторінці не знайдено.")
            break

        print(f"✅ Знайдено {len(replays)} матчів. Перевіряю...")

        for replay in replays:
            if downloaded_count >= TARGET_DOWNLOADS:
                break # Ми досягли ліміту, який ти вказав!

            replay_id = replay['id']
            raw_title = replay.get('title', f"Match_{replay_id}")
            if raw_title is None:
                raw_title = f"Match_{replay_id}"
                
            replay_title = raw_title.replace("/", "-").replace("\\", "-").replace(":", "").replace("?", "")
            file_path = f"rlcs_replays/{replay_title}_{replay_id}.replay"

            # Якщо файл вже є — просто пропускаємо і не зараховуємо як "нове скачування"
            if os.path.exists(file_path):
                print(f"⏩ Вже є на ПК: {replay_title}")
                continue

            print(f"[{downloaded_count + 1}/{TARGET_DOWNLOADS}] 📥 Завантажую: {replay_title}...")
            dl_url = f"https://ballchasing.com/api/replays/{replay_id}/file"
            
            # 3. Цикл безпечного завантаження (з обробкою помилки 429)
            retry = True
            while retry:
                dl_res = requests.get(dl_url, headers=headers)

                if dl_res.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(dl_res.content)
                    
                    downloaded_count += 1
                    retry = False
                    time.sleep(2) # Ввічлива пауза
                    
                elif dl_res.status_code == 429:
                    print("⚠️ Сервер просить пригальмувати (Помилка 429). Чекаємо 10 секунд і пробуємо знову...")
                    time.sleep(10)
                else:
                    print(f"❌ Помилка завантаження {replay_id}: {dl_res.status_code}")
                    retry = False # Якщо помилка інша (наприклад 404) - пропускаємо файл

        # 4. Перехід на наступну сторінку та ЗБЕРЕЖЕННЯ ПАМ'ЯТІ
        next_url = data.get('next')
        if next_url:
            url = next_url
            with open(STATE_FILE, "w") as f:
                f.write(url)
            print(f"💾 Прогрес збережено. Готовий до наступної сторінки.")
        else:
            print("🏁 Це була остання сторінка на Ballchasing. Більше реплеїв за цим фільтром немає!")
            url = None
            if os.path.exists(STATE_FILE):
                os.remove(STATE_FILE) # Очищаємо пам'ять, бо качати більше нічого

    elif response.status_code == 429:
        print("⚠️ Ліміт запитів до API при гортанні списку (Помилка 429). Чекаємо 15 секунд...")
        time.sleep(15)
    else:
        print(f"❌ Помилка доступу до списку: {response.status_code} - {response.text}")
        break

print(f"\n🎉 Роботу скрипта завершено! Успішно скачано {downloaded_count} НОВИХ реплеїв.")