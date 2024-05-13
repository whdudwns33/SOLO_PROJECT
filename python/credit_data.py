import pandas as pd
import statsmodels.api as sm

# 데이터 파일 읽기
df = pd.read_csv("credit_data.csv", low_memory=False, na_values='*')
# print("df:", df)

# 더미 변수 생성
dummy = pd.get_dummies(df['LIV_ADD'], prefix='LIV_ADD', dtype=float)
# print("dummy:", dummy)

# 두 데이터프레임을 열 방향으로 합치기
combined_df = df.join(dummy)
# print("combined_df:", combined_df)


number_columns = combined_df.select_dtypes(include=["number"])
# print("number_columns:", number_columns)

# 결측치 제거
final_df = number_columns.fillna(0)
# print("final_df:", final_df)

# 종속 변수 선택
y = df["CB"].astype(float)

# 독립 변수 선택
# 1. 지역 변수 기반
# X = dummy

# 2. 전체 데이터 기반
# y 값 및
X = final_df.drop(columns=["CB", "SP", "식별구분", "결과값(연체회차)"]).astype(float)

# 모든 독립 변수에 상수항 추가
X = sm.add_constant(X)
# 모델 생성
model = sm.OLS(y, X)
# 모델 피팅
result = model.fit()
# print(f"SUMMARY : {result.summary()}")


# 후진 제거법 활용
# 모든 데이터를 활용하여 OLS 모델을 설계했기 때문에 p-value로 유의한 데이터셋만 남기도록 반복
# 독립변수 리스트화
# 초기 변수 선택
selected_features = X.columns.tolist()

# 후진 제거법을 통한 변수 선택
while len(selected_features) > 1:  # 최소 하나의 변수를 남길 때까지 반복
    # 현재 선택된 변수로 모델 생성
    X_subset = X[selected_features]
    model = sm.OLS(y, X_subset)
    result = model.fit()

    # 가장 큰 p-value를 가진 변수 제거
    max_p_value = result.pvalues.drop("const").max()
    if max_p_value > 0.05:  # 예를 들어 0.05의 유의수준을 가질 때
        max_p_feature = result.pvalues.drop("const").idxmax()
        selected_features.remove(max_p_feature)
        print(f"Removing feature '{max_p_feature}' with p-value {max_p_value:.4f}")
    else:
        break

# 최종 모델 결과 출력
final_model = sm.OLS(y, X[selected_features])
final_result = final_model.fit()
print(f"FINAL SUMMARY: {final_result.summary()}")