import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# scikit-learn에서 데이터 스케일링을 위한 라이브러리 임포트
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# 1. 데이터 불러오기: 첨부된 minmax scaling 파일
data_path = 'data/Social_outlier_removed.csv'
df = pd.read_csv(data_path)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])


print("데이터 크기:", df.shape)
print("컬럼 목록:", df.columns.tolist())

# 2. 피처와 타깃 변수 분리
# 여기서는 'Purchased' 컬럼이 예측할 타깃 변수라고 가정합니다.
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# 3. 학습과 테스트 데이터 셋 분리 (예: 70% 학습, 30% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train1 = X_train.drop(['Gender','User ID'],axis=1)
X_test1 = X_test.drop(['Gender','User ID'],axis=1)
scaler_minmax = MinMaxScaler()
X_train_scaled = scaler_minmax.fit_transform(X_train1)
X_test_scaled = scaler_minmax.transform(X_test1)


# 4. 로지스틱 회귀 모델 초기화
# API에 명시된 파라미터를 그대로 사용합니다.
model = LogisticRegression(
    penalty='l2',               # L2 규제, none이면 
    dual=False,
    tol=0.0001,
    C=10,                      # 규제 강도
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=42,
    solver='lbfgs',
    max_iter=500,               # 최대 반복수 
    multi_class='deprecated',
    verbose=0,                  # 설명 안 한다 
    warm_start=False,
    n_jobs=None,
    l1_ratio=None               # 엘라스틱넷 쓰지 않아서 필요 X
)

# 5. 모델 학습
model.fit(X_train_scaled, y_train)

# 6. 테스트 데이터에 대한 예측 수행
y_pred = model.predict(X_test_scaled)

# 7. 예측 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test,y_pred)
class_report = classification_report(y_test, y_pred)


print("정확도(Accuracy): {:.2f}".format(accuracy))
print("혼동 행렬(Confusion Matrix):")
print(conf_matrix)
print('ROC_AUC()')
print(f'{roc_auc}')
print("분류 보고서(Classification Report):")
print(class_report)