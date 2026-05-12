import pandas as pd
import joblib

print("Завантажую PRO-мозок ШІ (V2) та реплей...")
df = pd.read_csv('ai_features_dataset_v2.csv')
model = joblib.load('ai_coach_model_v2.pkl')

# --- ФІКС: Додали всі 8 параметрів, які чекає нейромережа ---
features = [
    'Ball_Y', 
    'Nearest_Teammate', 
    'Nearest_Opponent', 
    'Is_Last_Man', 
    'Boost_Amount', 
    'Player_Speed', 
    'Player_VY', 
    'Player_Y'
]

# predict_proba видає ймовірність від 0.0 до 1.0 (шанс зберегти м'яч)
# Чим нижчий відсоток, тим тупішим було рішення лізти на м'яч
df['AI_Expected_Possession'] = model.predict_proba(df[features])[:, 1] * 100

# Фільтруємо і шукаємо найгірші рішення (де шанс зберегти м'яч був < 25%)
mistakes = df[df['AI_Expected_Possession'] < 25].sort_values(by='AI_Expected_Possession')

print("\n🚨 ТОП-5 НАЙГІРШИХ РІШЕНЬ (Bad Challenges):")
for idx, row in mistakes.head(5).iterrows():
    print(f"[{row['Time']} сек] Гравець: {row['Player']}")
    print(f"   -> ШІ оцінив шанс успіху лише у {row['AI_Expected_Possession']:.1f}%")
    print(f"   -> Позиція: Тімейт був за {row['Nearest_Teammate']} од., суперник за {row['Nearest_Opponent']} од.")
    print(f"   -> Ресурси: Буст {row['Boost_Amount']}%, Швидкість {row['Player_Speed']} од.")
    print(f"   -> Реальність: М'яч {'збережено дивом!' if row['Possession_Kept'] == 1 else 'ВТРАЧЕНО (як ШІ і казав)'}\n")

# Шукаємо геніальні рішення (шанс > 80%)
good_plays = df[df['AI_Expected_Possession'] > 80].sort_values(by='AI_Expected_Possession', ascending=False)
print("✅ ТОП-5 НАЙБЕЗПЕЧНІШИХ ДІЙ (Smart Plays):")
for idx, row in good_plays.head(5).iterrows():
    print(f"[{row['Time']} сек] {row['Player']} зіграв ідеально (Шанс успіху {row['AI_Expected_Possession']:.1f}%)")
    print(f"   -> Ресурси: Буст {row['Boost_Amount']}%, Швидкість {row['Player_Speed']} од.\n")