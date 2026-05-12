import os
import requests
import time

API_KEY = "cTWL9ikjuQ7Pyu2fIfnDwtM9n7PYZLcG8YtjTBzp"

# Створюємо папку для реплеїв
os.makedirs("rlcs_replays", exist_ok=True)
headers = {'Authorization': API_KEY}

# 2. Твій ідеальний фільтр перекладений для API
# title=rlcs, pro=true (тільки про-гравці), min-rank=supersonic-legend (SSL)
print("🔍 Шукаю RLCS реплеї рівня SSL (Supersonic Legend)...")

# Беремо 50 найкращих матчів (можеш збільшити count до 100 або 200, якщо треба більше)
url = "https://ballchasing.com/api/replays?title=rlcs&pro=true&min-rank=supersonic-legend&count=50"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    replays = response.json().get('list', [])
    print(f"✅ Знайдено {len(replays)} матчів. Починаю завантаження...\n")

    for i, replay in enumerate(replays):
        replay_id = replay['id']
        raw_title = replay.get('title', f"Match_{replay_id}")
        if raw_title is None:
            raw_title = f"Match_{replay_id}"
            
        replay_title = raw_title.replace("/", "-").replace("\\", "-").replace(":", "").replace("?", "")
        file_path = f"rlcs_replays/{replay_title}_{replay_id}.replay"

        if os.path.exists(file_path):
            print(f"[{i+1}/{len(replays)}] ⏩ Вже є на ПК: {replay_title}")
            continue

        print(f"[{i+1}/{len(replays)}] 📥 Завантажую: {replay_title}...")
        dl_url = f"https://ballchasing.com/api/replays/{replay_id}/file"
        
        # Робимо запит
        dl_res = requests.get(dl_url, headers=headers)

        if dl_res.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(dl_res.content)
            
            # ФІКС: Ввічлива пауза на 2 секунди після кожного скачування
            time.sleep(2)
            
        elif dl_res.status_code == 429:
            # Якщо сервер все одно свариться, робимо велику паузу
            print("⚠️ Сервер просить пригальмувати (Помилка 429). Чекаємо 10 секунд...")
            time.sleep(10)
        else:
            print(f"❌ Помилка завантаження {replay_id}: {dl_res.status_code}")
            
    print("\n🎉 Всі реплеї успішно завантажено в папку 'rlcs_replays'!")
else:
    print(f"❌ Помилка доступу до API: {response.status_code} - {response.text}")