package com.example.interest.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import java.util.List;
import java.util.Map;

@Service
public class OpenAIService {

    @Value("${OPENAI_API_KEY}")
    private String apiKey;

    @Value("${openai.model}")
    private String model;

    @Value("${openai.url}")
    private String apiUrl;

    private final WebClient webClient = WebClient.builder().build();

    public Map<String, Object> getResponse(String userMessage) {
        Map<String, Object> body = Map.of(
                "model", model,
                "messages", List.of(
                        Map.of("role", "system", "content",
                                "너는 사용자의 답변을 분석해서 5가지 주제(영화/미디어, 스포츠, 경제/재테크, 기술/IT, 일상/여행)의 신뢰도 퍼센티지를 JSON으로 응답해야 해."),
                        Map.of("role", "user", "content", userMessage)
                )
        );

        return webClient.post()
                .uri(apiUrl)
                .header("Authorization", "Bearer " + apiKey)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(Map.class)
                .block();
    }
}
