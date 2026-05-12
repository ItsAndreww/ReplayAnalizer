import pandas as pd
import joblib

print("Завантажую мозок ШІ та реплей...")
df = pd.read_csv('ai_features_dataset.csv')
model = joblib.load('ai_coach_model.pkl')

# Просимо ШІ оцінити кожен дотик у матчі
features = ['Ball_Y', 'Nearest_Teammate', 'Nearest_Opponent', 'Is_Last_Man']

# predict_proba видає ймовірність від 0.0 до 1.0 (шанс зберегти м'яч)
# Чим нижчий відсоток, тим тупішим було рішення лізти на м'яч
df['AI_Expected_Possession'] = model.predict_proba(df[features])[:, 1] * 100

# Фільтруємо і шукаємо найгірші рішення (де шанс зберегти м'яч був < 25%)
mistakes = df[df['AI_Expected_Possession'] < 25].sort_values(by='AI_Expected_Possession')

print("\n🚨 ТОП-5 НАЙГІРШИХ РІШЕНЬ У МАТЧІ (Bad Challenges):")
for idx, row in mistakes.head(5).iterrows():
    print(f"[{row['Time']} сек] Гравець: {row['Player']}")
    print(f"   -> ШІ оцінив шанс успіху лише у {row['AI_Expected_Possession']:.1f}%")
    print(f"   -> Чому? Тімейт був за {row['Nearest_Teammate']} юнітів, а суперник за {row['Nearest_Opponent']}.")
    print(f"   -> Реальність: М'яч {'збережено дивом!' if row['Possession_Kept'] == 1 else 'ВТРАЧЕНО (як ШІ і казав)'}\n")

# Шукаємо геніальні рішення (шанс > 80%)
good_plays = df[df['AI_Expected_Possession'] > 80].sort_values(by='AI_Expected_Possession', ascending=False)
print("✅ ТОП-5 НАЙБЕЗПЕЧНІШИХ ДІЙ (Smart Plays):")
for idx, row in good_plays.head(5).iterrows():
    print(f"[{row['Time']} сек] {row['Player']} зіграв ідеально (Шанс успіху {row['AI_Expected_Possession']:.1f}%)")