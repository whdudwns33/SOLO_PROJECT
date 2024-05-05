import pandas as pd
import statsmodels.api as sm
import numpy as np

# 데이터 파일 읽기
df = pd.read_csv("credit_data.csv", low_memory=False, na_values='*')
print("df:", df)

# 더미 변수 생성
dummy = pd.get_dummies(df['LIV_ADD'], prefix='LIV_ADD', dtype=float)
print("dummy:", dummy)

# 두 데이터프레임을 열 방향으로 합치기
combined_df = df.join(dummy)
print("combined_df:", combined_df)


number_columns = combined_df.select_dtypes(include=["number"])
print("number_columns:", number_columns)

# 결측치 제거
final_df = number_columns.fillna(0)
print("final_df:", final_df)

# 종속 변수 선택
y = df["CB"].astype(float)

# 독립 변수 선택
X = final_df.drop(columns=["CB", "SP", "식별구분", "결과값(연체회차)"]).astype(float)
# x의 결측값, 무한값 확인
# print("Nan : ",X.isnull().sum())
# print(f"inf : {np.isinf(X).sum()}")


# 모든 독립 변수에 상수항 추가
X = sm.add_constant(X)

print("Data types of y:", y.dtypes)
print("Data types of X:", X.dtypes)
# print("y values:", y.head())
# print("X values:", X.head())
# print("X head:", X.head())


# 모델 생성
model = sm.OLS(y, X)

# 모델 피팅
result = model.fit()

# AIC 값 확인
aic = result.aic
print(f"aci : {aic}")
