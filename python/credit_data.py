import pandas as pd
import statsmodels.api as sm

# 데이터 파일 읽기
df = pd.read_csv("credit_data.csv", low_memory=False)

# 원핫 인코딩 수행
one_hot_encoded = pd.get_dummies(df['LIV_ADD'], prefix='LIV_ADD')
print(one_hot_encoded)
# print(one_hot_encoded)

# # T OR F 형태의 데이터 수정
# one_hot_encoded_replace = one_hot_encoded.replace({'True': 1, 'False': 0})
# print(one_hot_encoded_replace)

# # 두 데이터프레임을 열 방향으로 합치기
# combined_df = pd.concat([df, one_hot_encoded], axis=1)
#
# print(combined_df.dtypes)
#
# # 종속 변수 선택
# y = combined_df["CB"]
#
# # 독립 변수 선택 (Y를 제외한 모든 변수)
# X = combined_df.drop(columns=["CB", "SP", "LIV_ADD","RES_ADD","ADD_YN"])
#
# # 모든 독립 변수에 상수항 추가
# X = sm.add_constant(X)
#
# # 모델 생성
# model = sm.OLS(y, X)
#
# # 모델 피팅
# result = model.fit()
#
# # AIC 값 확인
# aic = result.aic
