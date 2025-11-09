package com.example.interest.service;

import com.example.interest.domain.InterestResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;

@Service
public class InterestService {


    // ✅ application.yml 값 주입 (Render 환경변수 or 기본값)
    @Value("${python.ai.server.url}")
    private String flaskBaseUrl;


    public InterestResult analyze(List<String> answers) {

        // 기본 점수
        Map<String, Double> scores = new HashMap<>();
        scores.put("기술/IT", 0.0);
        scores.put("스포츠", 0.0);
        scores.put("영화/미디어", 0.0);
        scores.put("경제/재테크", 0.0);
        scores.put("일상/여행", 0.0);

        // ✅ 질문 키워드에 따라 점수 증가
        for (String ans : answers) {
            String lower = ans.toLowerCase();

            if (lower.contains("it") || lower.contains("컴퓨터") || lower.contains("기술") || lower.contains("개발")) {
                scores.put("기술/IT", scores.get("기술/IT") + 20);
            } else if (lower.contains("운동") || lower.contains("축구") || lower.contains("농구")) {
                scores.put("스포츠", scores.get("스포츠") + 20);
            } else if (lower.contains("영화") || lower.contains("드라마") || lower.contains("유튜브")) {
                scores.put("영화/미디어", scores.get("영화/미디어") + 20);
            } else if (lower.contains("주식") || lower.contains("부동산") || lower.contains("경제")) {
                scores.put("경제/재테크", scores.get("경제/재테크") + 20);
            } else {
                scores.put("일상/여행", scores.get("일상/여행") + 20);
            }
        }

        // 총합으로 퍼센티지 계산
        double total = scores.values().stream().mapToDouble(Double::doubleValue).sum();
        for (String key : scores.keySet()) {
            scores.put(key, (scores.get(key) / total) * 100);
        }

        // 내림차순 정렬 후 상위 2개 선택
        List<Map.Entry<String, Double>> sorted = new ArrayList<>(scores.entrySet());
        sorted.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        String top1 = sorted.get(0).getKey();
        String top2 = sorted.get(1).getKey();

        return new InterestResult(scores, top1, top2);
    }

    public Map<String, Double> analyzeWithPython(String userInput) {
        RestTemplate restTemplate = new RestTemplate();

        // Render 환경변수에서 읽은 기본 URL + predict 경로 조합
        String apiUrl = flaskBaseUrl + "/predict";

        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("sentence", userInput);

        try {
            ResponseEntity<Map> response = restTemplate.postForEntity(apiUrl, requestBody, Map.class);

            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                Map<String, Object> body = response.getBody();
                return (Map<String, Double>) body.get("scores"); // Flask JSON에서 scores 추출
            } else {
                throw new RuntimeException("Flask 응답 실패");
            }

        } catch (Exception e) {
            e.printStackTrace();
            // ❌ Flask 실패 시 임시 데이터
            Map<String, Double> fallback = new HashMap<>();
            fallback.put("영화/미디어", 40.0);
            fallback.put("스포츠", 25.0);
            fallback.put("경제/재테크", 15.0);
            fallback.put("기술/IT", 10.0);
            fallback.put("일상/여행", 6.0);
            return fallback;
        }
    }


}


