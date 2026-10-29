import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs('models', exist_ok=True)

print("1. 데이터 파일 불러오는 중...")
df = pd.read_excel('data/gangneung_data.xlsx')

# 2. 현실성 반영을 위한 확장 피처 선정
required_cols = ['기상상태', '노면상태', '주야', '도로형태', '사고유형', '피해운전자 차종', '피해운전자 연령대', '피해운전자 상해정도']
df_clean = df.dropna(subset=required_cols).copy()

# 타겟: 중상 이상(중상/사망)=1, 경상 이하=0
df_clean['중상이상여부'] = df_clean['피해운전자 상해정도'].apply(lambda x: 1 if str(x) in ['중상', '사망'] else 0)

features = ['기상상태', '노면상태', '주야', '도로형태', '사고유형', '피해운전자 차종', '피해운전자 연령대']
X = df_clean[features].astype(str).copy()
y = df_clean['중상이상여부']

print("2. 데이터 인코딩 및 학습 진행 중...")
encoders = {}
for col in features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 모델 및 인코더 저장
joblib.dump(model, 'models/accident_risk_model.pkl')
joblib.dump(encoders, 'models/label_encoders.pkl')

print("✅ 고도화된 AI 모델 학습 완료!")
