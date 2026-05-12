import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("1. Завантажую PRO-датасет (V2)...")
df = pd.read_csv('ai_features_dataset_v2.csv')

# --- НОВИЙ НАБІР ФІЧЕЙ (GAME SENSE) ---
features = [
    'Ball_Y',             # Де знаходиться м'яч
    'Nearest_Teammate',   # Детектор дабл-комітів
    'Nearest_Opponent',   # Детектор тиску
    'Is_Last_Man',        # Чи ти останній в захисті
    'Boost_Amount',       # Детектор жадібності (0-100 бусту)
    'Player_Speed',       # Детектор зупинки (збереження імпульсу)
    'Player_VY',          # Напрямок ротації
    'Player_Y'            # Глибина позиції гравця
]

X = df[features]
y = df['Possession_Kept']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Даних для навчання: {len(X_train)} ситуацій. Для екзамену: {len(X_test)} ситуацій.")
print("2. Треную нейромережу (XGBoost Pro)...")

# ПРОКАЧАНІ ПАРАМЕТРИ
model = XGBClassifier(
    n_estimators=200,      # Більше дерев
    learning_rate=0.05,    # Менший крок навчання (вчиться повільніше, але якісніше)
    max_depth=5,           # Глибина логіки
    subsample=0.8,         # Захист від зазубрювання (бере 80% випадкових ситуацій)
    colsample_bytree=0.8,  # Захист від зазубрювання (бере 80% випадкових фічей)
    random_state=42
)

# Навчаємо з підгляданням у тестовий набір (eval_set)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False  # Щоб не спамило кожен крок у консоль
)

print("3. Складаємо екзамен...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"🎯 Загальна точність передбачень ШІ: {accuracy * 100:.1f}%\n")

# НОВЕ: Аналізуємо, чи ШІ реально розуміє гру, чи просто вгадує більшість
print("📊 Детальний звіт розуміння гри (Classification Report):")
print(classification_report(y_test, predictions, target_names=['Втрата м\'яча (0)', 'Збереження (1)']))
print("-" * 50)

importances = model.feature_importances_
print("\n🧠 Що найважливіше для успіху (Пріоритети ШІ):")

feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances * 100})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

for idx, row in feature_importance_df.iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.1f}%")

# Зберігаємо новий PRO-мозок
joblib.dump(model, 'ai_coach_model_v2.pkl')
print("\n[Успіх] Прокачаний мозок ШІ збережено у файл 'ai_coach_model_v2.pkl'!")