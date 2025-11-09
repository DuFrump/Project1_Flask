import joblib
from konlpy.tag import Okt

print("주제 예측 프로그램을 시작합니다.")

# --- 2. 학습할 때와 동일한 토크나이저 함수 정의 ---
okt = Okt()
def tokenize(text):
    return [word for word, pos in okt.pos(text, stem=True) if pos in ['Noun', 'Verb', 'Adjective']]

# --- 1. 학습된 모델과 필요 도구 불러오기 ---
try:
    # 1단계에서 저장했던 파일들을 불러옵니다.
    model = joblib.load('./Project1/models/LightGBM/korean_topic_model.pkl')
    vectorizer = joblib.load('./Project1/models/LightGBM/tfidf_vectorizer.pkl')
    label_encoder = joblib.load('./Project1/models/LightGBM/label_encoder.pkl')
    print("모델과 도구들을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 저장된 모델 파일(korean_topic_model.pkl)이나 관련 도구 파일이 없습니다.")
    print("먼저 XGBoostClassifier.py 파일을 실행하여 모델을 학습하고 저장해주세요.")
    exit() # 프로그램 종료

# --- 3. 사용자 입력 및 예측 무한 반복 ---
while True:
    print("-" * 50)
    # 사용자로부터 문장을 입력받습니다.
    user_input = input("분석할 문장을 입력하세요 (종료하려면 'exit' 입력): ")

    # 'exit'를 입력하면 프로그램을 종료합니다.
    if user_input.lower() == 'exit':
        print("프로그램을 종료합니다. 이용해주셔서 감사합니다.")
        break
    
    # 입력된 문장이 비어있으면 다시 입력받습니다.
    if not user_input.strip():
        print("문장을 입력해주세요.")
        continue

    # --- 4. 예측 수행 ---
    # 1. 입력된 텍스트를 TF-IDF 벡터로 변환 (학습 때 썼던 vectorizer 사용)
    #    - 주의: fit_transform이 아니라 transform을 사용해야 합니다.
    input_vec = vectorizer.transform([user_input])

    # 2. 벡터를 모델에 넣어 예측 (결과는 숫자로 나옴)
    predicted_label_encoded = model.predict(input_vec)

    # 3. 예측된 숫자를 다시 우리가 아는 주제(문자열)로 변환
    predicted_label = label_encoder.inverse_transform(predicted_label_encoded)

    # 4. 최종 결과 출력
    print(f"예측 결과: 이 문장은 **{predicted_label[0]}** 주제에 가깝습니다.")