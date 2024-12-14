from konlpy.tag import Okt
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from difflib import SequenceMatcher

# Step 1: 네이버 API 데이터 수집
def fetch_naver_data(query, api_url, client_id, client_secret, display=10):
    """네이버 API를 이용하여 데이터 수집"""
    headers = {
        "X-Naver-Client-Id": "3K8lZhOl84EithJ3EWIb",
        "X-Naver-Client-Secret": "80yNaLWC1m"
    }
    params = {
        "query": query,
        "display": display,
        "start": 1
    }

    try:
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()  # 오류 발생 시 예외 발생
        data = response.json()
        return [item['title'] + " " + item['description'] for item in data['items']]
    except Exception as e:
        print(f"Error fetching data from Naver API: {e}")
        return []

# Step 2: 데이터 전처리
def preprocess_text(texts):
    """간단한 전처리: 특수문자 제거 및 소문자 변환"""
    processed_texts = []
    for text in texts:
        text = re.sub(r"[^\w\s가-힣]", "", text)  # 특수문자 제거
        processed_texts.append(text)
    return processed_texts

# Step 3: 텍스트를 띄어쓰기로 분리
def split_by_whitespace(texts):
    """입력된 텍스트를 띄어쓰기로 분리하여 단어 리스트로 반환"""
    result_list = []
    for text in texts:
        separated_words = text.split()
        result_list.extend(separated_words)
    return result_list

# Step 4: TF-IDF 기반 신조어 탐지
def find_neologisms_tfidf(texts, min_tfidf_score=0.1):
    """TF-IDF로 신조어 후보를 탐지"""
    vectorizer = TfidfVectorizer(
        min_df=2,  # 최소 빈도: 2로 설정
        max_df=0.85,  # 최대 빈도: 문서의 85% 이하
        token_pattern=r'\b[가-힣]+\b'  # 한글 단어만 추출
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        words = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1

        word_scores = [(word, score) for word, score in zip(words, tfidf_scores) if score >= min_tfidf_score]
        return [word for word, score in sorted(word_scores, key=lambda x: x[1], reverse=True)]
    except ValueError as e:
        print("TF-IDF Error:", e)
        return []  # 오류 발생 시 빈 리스트 반환

# Step 5: 신조어 필터링
def filter_neologisms_advanced(tfidf_candidates):
    okt = Okt()

    # 불용어 리스트
    stopwords = [
        "뜻", "부동산", "영어", "요즘", "그", "대해", "아파트", "관련", "그리고", 
        "뜻에", "실제", "오늘은", "유래", "있습니다", "자주", "최근", "공연", 
        "국회", "단어", "많이", "사례", "사용", "사이에서", "안녕하세요", "영어로", 
        "운동을", "유사한", "유행하는", "의미를", "의미하는", "정신적", "정확히", 
        "필요한", "합니다", "합친", "흔히", "용어", "인기", "조국", "청약", "가격"
    ]

    filtered_words = []
    for word in tfidf_candidates:
        # 불용어 제거
        if word in stopwords:
            continue

        # 품사 분석: 명사만 포함
        pos_tags = okt.pos(word)
        if not all(tag == "Noun" for _, tag in pos_tags):
            continue

        # 신조어 패턴 확인
        if len(word) >= 3:
            filtered_words.append(word)

    return filtered_words

# 신조어 검색 및 캐싱
def search_neologism_meaning(word):
    """신조어의 의미를 검색하여 반환"""
    client_id = "3K8lZhOl84EithJ3EWIb"
    client_secret = "80yNaLWC1m"
    url = "https://openapi.naver.com/v1/search/encyc.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    params = {"query": word, "display": 1}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        print("API Response:", data)  # 응답 데이터 확인
        if "items" in data and len(data["items"]) > 0:
            meaning = data["items"][0]["description"]
            return meaning.strip()
        else:
            return "검색 결과 없음"
    except Exception as e:
        return f"에러 발생: {e}"

neologism_cache = {}

def get_neologism_meaning(word):
    """신조어의 의미를 캐싱하여 반환"""
    if word in neologism_cache:
        return neologism_cache[word]

    meaning = search_neologism_meaning(word)
    neologism_cache[word] = meaning  # 캐싱
    return meaning


def is_correct_answer(user_answer, correct_answer, threshold=0.6):
    """유사도 기반 정답 인정"""
    similarity = SequenceMatcher(None, user_answer, correct_answer).ratio()
    percentage = similarity * 100
    print(f"정답 유사성 : {percentage:.1f}%")
    return similarity >= threshold

# 문해력 테스트
def test_neologism_knowledge_with_dynamic_search(neologisms):
    """동적으로 신조어의 의미를 검색하고 문해력 테스트 진행"""
    print("신조어 문해력 테스트를 시작합니다!")
    score = 0
    total_questions = 3  # 질문 개수

    for i in range(total_questions):
        word = random.choice(neologisms)
        meaning = get_neologism_meaning(word)

        if meaning == "검색 결과 없음" or "에러 발생" in meaning:
            print(f"\n문제 {i + 1}: '{word}'의 검색 결과를 찾을 수 없습니다. 다음 문제로 넘어갑니다.")
            continue

        print(f"\n문제 {i + 1}: '{word}'의 뜻은 무엇일까요?")
        user_answer = input("정답을 입력하세요: ").strip()

        if is_correct_answer(user_answer, meaning):
            print("정답입니다! 🎉")
            score += 1
        else:
            print(f"오답입니다. 정답은: '{meaning}'입니다.")

    print(f"\n테스트 종료! 총 점수: {score}/{total_questions}")


# Main 파이프라인
def main():
    # 네이버 API 설정
    client_id = "3K8lZhOl84EithJ3EWIb"  # 네이버 Client ID
    client_secret = "80yNaLWC1m"  # 네이버 Client Secret

    # API URL 설정
    blog_api_url = "https://openapi.naver.com/v1/search/blog.json"
    news_api_url = "https://openapi.naver.com/v1/search/news.json"

    # 데이터 수집
    query = "신조어"  # 검색 키워드
    blog_data = fetch_naver_data(query, blog_api_url, client_id, client_secret, display=10)
    news_data = fetch_naver_data(query, news_api_url, client_id, client_secret, display=10)

    # Step 1: 데이터 준비
    raw_texts = blog_data + news_data

    # Step 2: 전처리
    processed_texts = preprocess_text(raw_texts)

    # Step 3: 명사 추출
    words = split_by_whitespace(processed_texts)
    print("split_by_whitespace:", words)
    print("-" * 80)

    # Step 4: TF-IDF로 신조어 후보 탐지
    tfidf_candidates = find_neologisms_tfidf(words)
    print("TF-IDF Based Candidates:", tfidf_candidates)
    print("-" * 80)

    filtered_neologisms = filter_neologisms_advanced(tfidf_candidates)
    print("Filtered Neologisms:", filtered_neologisms)
    print("-" * 80)

    # 문해력 테스트 실행
    test_neologism_knowledge_with_dynamic_search(filtered_neologisms)

if __name__ == "__main__":
    main()

