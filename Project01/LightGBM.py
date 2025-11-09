import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os 
from collections import Counter
from konlpy.tag import Okt
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# --- 폰트 설정 (기존과 동일) ---
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------------
# 1. 데이터 로드 및 통합 (기존과 동일)
# -------------------------------------------------------------------
print("---------- 데이터 로드 및 통합 시작 ----------")

topics = ['기술/IT', '스포츠', '영화/미디어', '경제/제테크', '일상/여행', 'NaN']
file_map = {
    '기술/IT': '기술_IT.txt', 
    '스포츠': '스포츠.txt', 
    '영화/미디어': '영화_미디어.txt', 
    '경제/제테크': '경제_제테크.txt', 
    '일상/여행': '일상_여행.txt',
    'NaN' : 'NaN.txt'
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
# 2. 데이터 분할 (기존과 동일)
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
        my_stopwords = [line.strip() for line in f if line.strip()]
    print(f"불용어 {len(my_stopwords)}개를 파일에서 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"경고: 불용어 파일('{stopwords_path}')을 찾을 수 없습니다. 불용어 처리 없이 진행합니다.")
    my_stopwords = []

vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=0.80, min_df=2, stop_words=my_stopwords, ngram_range=(1, 2))

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
        '펀드', '환율', 'ETF', '코스피'
    ],
    '기술/IT': [
        '네트워크', '데이터', '드라이버', '반도체', '서버', '소프트웨어',
        '스마트폰', '알고리즘', '어플', '업데이트', '인공지능', '코딩',
        '프로그래밍', '하드웨어', 'AI', 'CPU', 'GPU'
    ],
    '영화/미디어': [
        '감독', '개봉', '극장', '넷플릭스', '드라마', '리뷰', '배우', '시나리오',
        '애니메이션', '연기', '예고편', '영화', '작품', '주연', '촬영', 'OTT', '구독료', '다큐멘터리'
    ],
    '일상/여행': [
        '항공권', '호텔', '휴가', '해외여행', '국내여행', '공항', '전시회',
        '콘서트', '반려동물', '요리', '여행', '카페', '맛집', '친구', 
        '주말', '취미', '산책'
    ]
}
KEYWORD_BOOST_WEIGHT = 2.5  # 가중치 값

print(f"\n핵심 키워드에 가중치(x{KEYWORD_BOOST_WEIGHT})를 적용합니다...")
for topic, keywords in keyword_weights.items():
    for keyword in keywords:
        try:
            keyword_index = vectorizer.vocabulary_[keyword]
            X_train_vec[:, keyword_index] *= KEYWORD_BOOST_WEIGHT
            X_val_vec[:, keyword_index] *= KEYWORD_BOOST_WEIGHT
        except KeyError:
            pass # 단어장에 없는 키워드는 통과

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
# 4. LightGBM 분류 모델 학습
# -------------------------------------------------------------------
lgbm_classifier = LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    reg_lambda=1.5,       # XGBoost의 reg_lambda와 같은 L2 규제 파라미터입니다.
    n_jobs=-1,
    random_state=42,
    verbose=2            # 학습 과정의 로그를 출력하지 않도록 설정합니다.
)

print("LightGBM 분류 모델 학습 시작...")
lgbm_classifier.fit(X_train_resampled, y_train_resampled)

# -------------------------------------------------------------------
# 5. 모델 성능 검증 및 예측
# -------------------------------------------------------------------
y_pred_encoded = lgbm_classifier.predict(X_val_vec) 
accuracy = accuracy_score(y_val_encoded, y_pred_encoded)
macro_f1 = f1_score(y_val_encoded, y_pred_encoded, average='macro')

print("\n------------------- 학습 및 검증 결과 -------------------")
print(f"최종 정확도 (Accuracy): {accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f} (클래스별 균형 성능)")
print("---------------------------------------------------------")
    
predicted_labels = label_encoder.inverse_transform(y_pred_encoded[:5])
print("상위 5개 예측 라벨 (복원):", predicted_labels) 
print("---------------------------------------------------------")

samples = [
    '아, 요즘 엔비디아랑 AMD 주식에 관심이 있어.',
    '요즘 스페인 축구 구단 FC 바르셀로나가 너무 못해서 마음이 아파..',
    '이번에 개봉된 영화 봤어? 재밌더라.',
    '물리학의 전반적인 역사에 대해 알고 싶어. 그게 관심이 가는 주제야.',
    '삼엽충 화석의 고생물학적 연구가 최근 고려대학교에서 발표되었대.',
    '미술관에서 조선시대 도자기 감상하는 게 취미인 사람이 많을까?',
    '혹시 말이야, 엔비디아의 새로운 GPU 드라이버 업데이트했어?',
    '난 요즘 유럽 축구 투어에 관심이 있어.',
    '난 요즘 분위기 좋은 카페 탐방과 여행에 관심이 있어.',
    '난 요즘 인공지능 프로그래밍에 관심이 있어.',
    '난 요즘 영화 보는 것에 푹 빠져있어.',
    '요즘 20대의 투자 방식에 대해 많은 것을 공부하려고 하고 있어.',
    '아침에 조깅하고 왔어. 달리기를 했는데 너무 재밌더라!',
    '요즘 미국 기술주 중심의 ETF에 적립식으로 투자하고 있어.',
    '이번 주말 K리그 경기는 정말 역대급으로 치열했어.',
    '나는 영화를 볼 때 감독의 전작들을 모두 찾아보는 습관이 있어.',
    '새로 출시된 그래픽카드 성능이 이전 세대보다 얼마나 향상되었는지 궁금하다.',
    '퇴근 후에 친구랑 동네 맛집에서 저녁 먹기로 약속했어.',
    '르네상스 시대 미술이 현대 디자인에 미친 영향에 대해 토론해 보고 싶어.',
    '그 선수가 구단과 연봉 협상에서 갈등을 빚고 있다는 기사를 봤어.',
    '요즘 OTT 구독료가 너무 올라서 어떤 서비스를 해지할지 고민 중이야.',
    '다음 달에 떠날 해외여행을 위해 환율을 매일 체크하고 있다.',
    '프로그래밍 언어의 역사와 발전 과정에 대한 다큐멘터리를 재밌게 봤어.',
    '헬스장에서 유산소 운동보다 근력 운동에 더 집중하려고 해.',
    '셰익스피어의 4대 비극이 인간의 보편적 감정을 어떻게 다루는지 분석해 볼까?',
    '난 이 스포츠 팀의 유니폼을 디자인한 디자이너의 팬이야.',
    '저 배우가 출연한 작품은 믿고 보는 편이라, 이번 신작도 바로 시청했지.',
    '최근 발표된 소비자물가지수가 예상보다 높아서 시장이 긴장하고 있대.',
    '주말에 아무 계획 없이 그냥 집에서 반려묘랑 뒹굴거리는 게 최고야.',
    '인공지능 윤리 문제는 기술 개발 속도를 따라가지 못하고 있는 것 같아.',
    'E-스포츠 구단의 운영 방식이 전통적인 스포츠 구단과 어떻게 다른지 알아보고 있어.',
    '분위기 좋은 카페에서 책 읽는 것만큼 좋은 재충전 방법도 없는 것 같아.',
    '고대 로마의 수도 시설이 당시 공중보건에 기여한 바가 크다고 생각해.'
]

samples_vec = vectorizer.transform(samples)
predicted_encoded = lgbm_classifier.predict(samples_vec)
predicted_labels = label_encoder.inverse_transform(predicted_encoded)

print("\n----------- LightGBM 문장 예측 결과 -----------")
for text, label in zip(samples, predicted_labels):
    print(f"'{text}'  ->  예측: **{label}**")
print("-------------------------------------------")

new_samples = [
    # 주제가 섞인 문장
    "E-스포츠 팀의 데이터 분석가가 되려고 파이썬 코딩을 배우고 있어.",
    "최근 개봉한 영화의 흥행 실패로 관련 미디어 기업의 주가가 하락했어.",
    "영화 '반지의 제왕' 촬영지인 뉴질랜드로 배낭여행을 떠나려고 항공권을 알아봤다.",
    "인기 축구 선수가 광고하는 최신 스마트폰의 카메라 성능이 그렇게 좋대.",
    "다음 투자처로 엔터테인먼트 산업의 성장 가능성을 분석하는 리포트를 읽었다.",

    # 핵심 키워드가 없는 미묘한 문장
    "이번 주말에 다 같이 유니폼 입고 경기장 가서 목청껏 소리 지르자.",
    "요즘 같은 시기에는 원금 보장되는 상품에 돈을 묶어두는 게 제일 안전해.",
    "그 배우의 차기작이 정말 기대돼. 이번엔 어떤 캐릭터를 연기할지 궁금하다.",
    "퇴근하고 동네 공원에서 잠깐 바람 쐬는 게 요즘 내 유일한 행복이야.",

    # 'NaN' 판별이 어려운 문장
    "머신러닝 모델의 과적합 문제를 해결하기 위한 수학적 원리가 궁금하다.",
    "조선시대 후기 민화에 나타난 해학적 표현 양식에 대해 토론해보자.",
    "애덤 스미스의 '국부론'이 현대 자본주의에 미친 영향은 무엇일까?",
    "고대 그리스 철학자들이 생각한 '행복'의 정의에 대해 논하시오."
]

new_samples_vec = vectorizer.transform(new_samples)
new_predicted_encoded = lgbm_classifier.predict(new_samples_vec)
new_predicted_labels = label_encoder.inverse_transform(new_predicted_encoded)

print("\n----------- LightGBM 새로운 문장 예측 결과 -----------")
for text, label in zip(new_samples, new_predicted_labels):
    print(f"'{text}'  ->  예측: **{label}**")
print("-------------------------------------------")

# -------------------------------------------------------------------
# 6. 학습된 모델 및 필요 도구 저장하기
# -------------------------------------------------------------------
import joblib # joblib 라이브러리를 임포트합니다.

print("\n학습된 모델과 도구들을 저장합니다...")

# 모델, 벡터라이저, 라벨 인코더를 각각 파일로 저장합니다.
# 이 3개는 예측할 때 세트로 필요해요!
joblib.dump(lgbm_classifier, './Project1/models/LightGBM/korean_topic_model.pkl')
joblib.dump(vectorizer, './Project1/models/LightGBM/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, './Project1/models/LightGBM/label_encoder.pkl')

print("모델과 도구들이 성공적으로 저장되었습니다. ✅")
print("이제 LightGBM_predict_topic.py 파일을 실행하여 예측을 시작할 수 있습니다.")