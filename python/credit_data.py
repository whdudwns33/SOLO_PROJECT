import pandas as pd
import statsmodels.api as sm

# 데이터 파일 읽기
df = pd.read_csv("credit_data.csv", low_memory=False, na_values='*')

# 원핫 인코딩 수행
dummy = pd.get_dummies(df['LIV_ADD'], prefix='LIV_ADD', dtype=float)
print(dummy.dtypes)

# # 두 데이터프레임을 열 방향으로 합치기
combined_df = pd.concat([df, dummy], axis=1)
non_numeric_columns = combined_df.select_dtypes(exclude=['number']).columns

print("Non-numeric columns:", non_numeric_columns)


# 종속 변수 선택
y = combined_df["CB"]

# 독립 변수 선택 (Y를 제외한 모든 변수)
X = combined_df.drop(columns=["CB", "SP", "LIV_ADD","RES_ADD","ADD_YN"])

# 모든 독립 변수에 상수항 추가
X = sm.add_constant(X)

print("Data types of y:", y.dtype)
print("Data types of X:", X.dtypes)
print("y values:", y.head())
print("X values:", X.head())
print("X type:", X.dtypes)
print("X head:", X.head())





# 모델 생성
model = sm.OLS(y, X)

# 모델 피팅
result = model.fit()

# AIC 값 확인
aic = result.aic
