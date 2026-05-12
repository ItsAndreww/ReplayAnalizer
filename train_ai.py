import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("1. Завантажую PRO-датасет (V2)...")
# Тепер ми читаємо новий файл з бустом і швидкістю!
df = pd.read_csv('ai_features_dataset_v2.csv')

# --- НОВИЙ НАБІР ФІЧЕЙ (GAME SENSE) ---
features = [
    'Ball_Y',             # Де знаходиться м'яч
    'Nearest_Teammate',   # Детектор дабл-комітів
    'Nearest_Opponent',   # Детектор тиску
    'Is_Last_Man',        # Чи ти останній в захисті
    'Boost_Amount',       # НОВЕ: Детектор жадібності (0-100 бусту)
    'Player_Speed',       # НОВЕ: Детектор зупинки (збереження імпульсу)
    'Player_VY',          # НОВЕ: Напрямок ротації
    'Player_Y'            # НОВЕ: Глибина позиції гравця
]

X = df[features]
y = df['Possession_Kept']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Даних для навчання: {len(X_train)} ситуацій. Для екзамену: {len(X_test)} ситуацій.")
print("2. Треную нейромережу (XGBoost Pro)...")

# Я трохи підняв глибину дерев (max_depth=5), бо тепер у нас більше складних фічей
model = XGBClassifier(
    n_estimators=150,      
    learning_rate=0.1,     
    max_depth=5,           
    random_state=42
)

# Магія навчання
model.fit(X_train, y_train)

print("3. Складаємо екзамен...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"🎯 Точність передбачень ШІ: {accuracy * 100:.1f}%\n")

importances = model.feature_importances_
print("📊 Що найважливіше для перемоги (на думку ШІ):")

# Сортуємо фічі за важливістю, щоб вивести гарний топ
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances * 100})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

for idx, row in feature_importance_df.iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.1f}%")

# Зберігаємо новий PRO-мозок
joblib.dump(model, 'ai_coach_model_v2.pkl')
print("\n[Успіх] Прокачаний мозок ШІ збережено у файл 'ai_coach_model_v2.pkl'!")