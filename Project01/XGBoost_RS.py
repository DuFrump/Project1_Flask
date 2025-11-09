import os 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from collections import Counter
from konlpy.tag import Okt
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# --- 폰트 설정 ---
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False


# -------------------------------------------------------------------
# 1. 데이터 로드 및 통합
# -------------------------------------------------------------------
print("---------- 데이터 로드 및 통합 시작 ----------")

topics = ['기술/IT', '스포츠', '영화/미디어', '경제/제테크', '일상/여행', '기타_주제']
file_map = {
    '기술/IT': '기술_IT.txt', 
    '스포츠': '스포츠.txt', 
    '영화/미디어': '영화_미디어.txt', 
    '경제/제테크': '경제_제테크.txt', 
    '일상/여행': '일상_여행.txt',
    '기타_주제' : '기타_주제.txt'
}

base_dir = './Project1/dataSet/created_dataset' 
X_train_text = []
y_train = []

for topic, file_name in file_map.items():
    file_path = os.path.join(base_dir, file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            X_train_text.extend(lines)
            y_train.extend([topic] * len(lines))
    except FileNotFoundError:
        print(f"경고: 파일 {file_name}을(를) 찾을 수 없습니다. 경로를 확인하세요.")
        continue  

print(f"총 학습 문장 개수: {len(X_train_text)}개")
print(f"라벨 분포: {Counter(y_train)}")
print("------------------------------------------")


# -------------------------------------------------------------------
# 2. 데이터 분할
# -------------------------------------------------------------------
X_train, X_val, y_train_labels, y_val_labels = train_test_split(
    X_train_text, y_train, test_size=0.2, random_state=42, stratify=y_train
)


# -------------------------------------------------------------------
# 3. 텍스트 벡터화 및 피처 엔지니어링
# -------------------------------------------------------------------
okt = Okt()

def tokenize(text):
    return [word for word, pos in okt.pos(text, stem=True) if pos in ['Noun', 'Verb', 'Adjective']]

# 불용어 처리
stopwords_path = './Project1/dataSet/stopwords-ko.txt'
try:
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        original_stopwords = [line.strip() for line in f if line.strip()]

        processed_stopwords = set()
        for stopword in original_stopwords:
            processed_stopwords.update(tokenize(stopword))
            processed_stopwords.add(stopword)
            
        my_stopwords = list(processed_stopwords)
        print(f"가공된 불용어 {len(my_stopwords)}개를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"경고: 불용어 파일('{stopwords_path}')을 찾을 수 없습니다. 불용어 처리 없이 진행합니다.")
    my_stopwords = []

vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=0.85, min_df=2, stop_words=my_stopwords, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

keyword_weights = {
    '스포츠': [
        '감독', '경기', '구단', '농구', '득점', '리그', '선수', '승리', '야구',
        '운동', '월드컵', '올림픽', '응원', '이적', '챔피언', '축구', '훈련'
    ],
    '경제/제테크': [
        '경제', '금리', '나스닥', '대출', '매도', '매수', '부동산', '세금', 
        '수익', '시장', '예금', '적금', '주가', '주식', '증시', '채권', '투자', 
        '펀드', '환율', 'ETF', '코스피', '기술주'
    ],
    '기술/IT': [
        '네트워크', '데이터', '드라이버', '반도체', '서버', '소프트웨어',
        '스마트폰', '알고리즘', '어플', '업데이트', '인공지능', '코딩',
        '프로그래밍', '하드웨어', 'AI', 'CPU', 'GPU', '해킹', '해커', '프로그래머',
        '개발자', '드라이버', '컴퓨터', '데스크탑', '노트북'
    ],
    '영화/미디어': [
        '감독', '개봉', '극장', '넷플릭스', '드라마', '리뷰', '배우', '시나리오',
        '애니메이션', '연기', '예고편', '영화', '작품', '주연', '촬영', 'OTT', '구독료', '다큐멘터리'
    ],
    '일상/여행': [
        '항공권', '호텔', '휴가', '해외여행', '국내여행', '공항', '전시회',
        '콘서트', '반려동물', '요리', '여행', '카페', '맛집', '친구', 
        '주말', '취미', '산책', '헬스장', '유산소', '근력'
    ]
}
KEYWORD_BOOST_WEIGHT = 3.0  # 가중치

print(f"\n핵심 키워드에 가중치(x{KEYWORD_BOOST_WEIGHT})를 적용합니다...")
for topic, keywords in keyword_weights.items():
    for keyword in keywords:
        try:
            keyword_index = vectorizer.vocabulary_[keyword]
            X_train_vec[:, keyword_index] *= KEYWORD_BOOST_WEIGHT
            X_val_vec[:, keyword_index] *= KEYWORD_BOOST_WEIGHT
        except KeyError:
            pass

# 라벨 인코딩
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_labels)
y_val_encoded = label_encoder.transform(y_val_labels)

print(f"학습 데이터셋 크기 (Features): {X_train_vec.shape}")
print("------------------------------------------")

# -------------------------------------------------------------------
# SMOTE 오버샘플링
# -------------------------------------------------------------------
print("\nSMOTE 오버샘플링 적용 시작...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train_encoded)
print(f"SMOTE 적용 후 학습 데이터: {X_train_resampled.shape}, 라벨: {y_train_resampled.shape}")
print(f"오버샘플링 후 라벨 분포: {Counter(y_train_resampled)}")
print("------------------------------------------")


# -------------------------------------------------------------------
# 4. 랜덤 서치를 이용한 최적 하이퍼파라미터 탐색
# -------------------------------------------------------------------
print("\n랜덤 서치를 이용한 최적 하이퍼파라미터 탐색을 시작합니다...")

# 4-1. 탐색할 파라미터 후보들을 정의합니다.
param_dist = {
    'n_estimators': [1000, 1200, 1500], # learning_rate가 낮아질 수 있으니 트리는 더 많이
    'max_depth': [5, 6],           # 최대 깊이를 낮춰 복잡도 제한
    'learning_rate': [0.05, 0.1],     # 너무 낮은 학습률은 제외
    'gamma': [0.4, 0.6, 0.8],         # 분기 제어 강화
    'reg_lambda': [3.0, 4.0, 5.0],     # 가중치 규제 강화
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

# 4-2. 기본 XGBoost 모델을 정의합니다.
xgb_base = XGBClassifier(
    n_jobs=-1,
    random_state=42,
    eval_metric='mlogloss',
)

# 4-3. RandomizedSearchCV 객체를 생성합니다.
# n_iter: 랜덤하게 50개의 조합을 시도합니다.
# cv: 3-fold 교차 검증을 사용합니다.
# scoring: 평가지표는 f1_macro를 사용합니다.
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=25,
    cv=3,
    scoring='f1_macro',
    verbose=2, # 진행 과정을 보여줍니다.
    random_state=42,
    n_jobs=-1
)

# 4-4. SMOTE로 증강된 데이터로 학습을 시작합니다.
random_search.fit(X_train_resampled, y_train_resampled)

# 4-5. 최적의 파라미터와 최고 점수를 출력합니다.
print("\n----------- 랜덤 서치 결과 -----------")
print(f"최적의 파라미터: {random_search.best_params_}")
print(f"최고 교차 검증 점수 (F1-Macro): {random_search.best_score_:.4f}")
print("------------------------------------")

# 4-6. 찾은 최적의 모델을 최종 모델로 사용합니다.
best_model = random_search.best_estimator_


# -------------------------------------------------------------------
# 5. '최적 모델'로 성능 검증 및 예측
# -------------------------------------------------------------------
y_pred_encoded = best_model.predict(X_val_vec)
accuracy = accuracy_score(y_val_encoded, y_pred_encoded)
macro_f1 = f1_score(y_val_encoded, y_pred_encoded, average='macro')

y_train_pred_encoded = best_model.predict(X_train_resampled)
train_accuracy = accuracy_score(y_train_resampled, y_train_pred_encoded)

print("\n------------------- 최종 모델 성능 -------------------")
print(f"학습 데이터 정확도: {train_accuracy:.4f}")
print(f"검증 데이터 정확도 (Accuracy): {accuracy:.4f}")
print(f"검증 데이터 Macro F1-Score: {macro_f1:.4f}")
print("---------------------------------------------------------")