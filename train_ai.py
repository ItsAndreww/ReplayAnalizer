import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

print("1. Завантаження V4 датасету (З підтримкою Score_Diff, Time_Remaining та 13 класів)...")
df = pd.read_csv('ai_features_dataset_v4.csv')

TARGET_COL = 'Action_Label'
EXCLUDE_COLS = ['Time', 'Player', 'Team', 'Action_Label', 'Action_Name']
FEATURES = [col for col in df.columns if col not in EXCLUDE_COLS]

# ДІАГНОСТИКА — покаже точний список фіч і їх кількість
print(f"\nВсього фіч: {len(FEATURES)}")
print("\nСписок всіх фіч по порядку:")
for i, f in enumerate(FEATURES):
    print(f"  {i+1:2d}. {f}")

X = df[FEATURES].fillna(0)
y = df[TARGET_COL]

label_mapping = df.drop_duplicates('Action_Label').set_index('Action_Label')['Action_Name'].to_dict()

print(f"Всього ситуацій: {len(df)}")
print(f"Кількість задіяних фічей: {len(FEATURES)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

from imblearn.over_sampling import SMOTE

print("\nЗастосування SMOTE для балансування міноритарних класів...")
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_resampled)

print("\n2. Тренування XGBoost...")

model = XGBClassifier(
    n_estimators=300,            
    learning_rate=0.03,          
    max_depth=7,                 
    max_bin=128,                 
    subsample=0.85,
    colsample_bytree=0.7,        
    min_child_weight=3,          
    gamma=0.2,                   
    reg_alpha=1.0,               
    reg_lambda=2.0,              
    tree_method='hist',
    early_stopping_rounds=40,    
    eval_metric="mlogloss",
    objective="multi:softprob",
    random_state=42,
    n_jobs=-1,  
)

X_train_np = X_train_resampled.to_numpy()
X_test_np = X_test.to_numpy()

model.fit(
    X_train_np, y_train_resampled,  # Використовуємо збалансований вектор цільової змінної
    sample_weight=sample_weights,
    eval_set=[(X_train_np, y_train_resampled), (X_test_np, y_test)],  # Оновлено y_train для валідації тренувального набору
    verbose=100
)

print(f"\nМодель зупинилась на {model.best_iteration} ітерації.")

print("\n3. Оцінка моделі:")
preds = model.predict(X_test_np)

sorted_labels = sorted(label_mapping.keys())
target_names = [label_mapping[i] for i in sorted_labels]

print(classification_report(y_test, preds, target_names=target_names, digits=3))

importances = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_ * 100
}).sort_values('Importance', ascending=False)

print("\nТоп-15 найважливіших фічей:")
print(importances.head(15).to_string(index=False))

joblib.dump(model, 'ai_coach_model_v4.pkl')
print("\nМодель збережено: ai_coach_model_v4.pkl")

print("\n4. Конвертація моделі у формат ONNX...")
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

update_registered_converter(
    XGBClassifier,
    "XGBoostXGBClassifier",
    calculate_linear_classifier_output_shapes,
    convert_xgboost,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)

initial_type = [('float_input', FloatTensorType([None, len(FEATURES)]))]
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={"zipmap": False},
    target_opset={"": 17, "ai.onnx.ml": 3}  
)

onnx_file_path = "ai_coach_v4.onnx"
with open(onnx_file_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"ONNX модель збережено: {onnx_file_path}")