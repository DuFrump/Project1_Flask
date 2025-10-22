from konlpy.tag import Okt

# Okt 객체를 여기서 한 번만 생성합니다.
okt = Okt()

def tokenize(text):
    """
    Okt 형태소 분석기를 사용해 명사, 동사, 형용사를 추출하는 함수
    """
    return [word for word, pos in okt.pos(text, stem=True) if pos in ['Noun', 'Verb', 'Adjective']]