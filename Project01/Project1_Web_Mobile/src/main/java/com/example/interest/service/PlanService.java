package com.example.interest.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;
import java.util.Map;

@Service
public class PlanService {

    private final WebClient webClient;

    @Value("${OPENAI_API_KEY}")
    private String apiKey;

    public PlanService() {
        this.webClient = WebClient.builder()
                .baseUrl("https://api.openai.com/v1/chat/completions")
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    /**
     * 🧩 1️⃣ 주제와 답변을 기반으로 일주일 계획 생성
     */
    public String generatePlan(String topic, List<String> answers) {
        String combinedAnswers = String.join(", ", answers);

        String prompt = """
            당신은 사용자의 관심사와 답변을 분석하여 맞춤형 일주일 계획을 제시하는 스마트 플래너입니다.
            주제: %s
            사용자의 답변 요약: %s

            월요일부터 일요일까지 각 요일별로 1~2문장으로 구체적인 계획을 작성하세요.
            (요일 출력은 <b>월요일</b> 이런 식으로 해줘)
            특히 별표(*), 밑줄(_), 또는 백틱(`)은 절대 포함하지 마세요.
            요일이나 제목은 단순히 줄바꿈으로 구분하세요.
            월요일 위쪽에 한칸 간격을 줘.
            
            """.formatted(topic, combinedAnswers);

        Map<String, Object> requestBody = Map.of(
                "model", "gpt-4o-mini",
                "messages", List.of(
                        Map.of("role", "system", "content", "당신은 일정 설계 전문가입니다."),
                        Map.of("role", "user", "content", prompt)
                )
        );

        try {
            Map<String, Object> response = webClient.post()
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
                    .bodyValue(requestBody)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();

            if (response == null || !response.containsKey("choices"))
                return "⚠️ OpenAI 응답이 비어 있습니다.";

            List<Map<String, Object>> choices = (List<Map<String, Object>>) response.get("choices");
            if (choices.isEmpty()) return "⚠️ choices 항목이 없습니다.";

            Map<String, Object> first = choices.get(0);
            Map<String, Object> message = (Map<String, Object>) first.get("message");

            if (message == null || !message.containsKey("content"))
                return "⚠️ message 내용이 없습니다.";

            return message.get("content").toString().replace("\n", "<br>");
        } catch (Exception e) {
            e.printStackTrace();
            return "⚠️ 계획을 생성하는 중 오류가 발생했습니다.";
        }
    }

    /**
     * 🧠 2️⃣ 기존 계획을 기반으로 사용자의 요청(변경 사항)을 반영하여 수정
     */
    public String refinePlan(String originalPlan, String userRequest) {
        String prompt = """
            당신은 일정 조정 전문가입니다.
            아래는 사용자가 이미 받은 일주일 계획입니다:
            %s

            사용자의 요청:
            "%s"

            위 계획의 형식을 유지하되, 사용자의 요청을 반영하여 수정된 버전의 일주일 계획을 제시하세요.
            (요일 출력은 <b>월요일</b> 이런 식으로 해줘)
            특히 별표(*), 밑줄(_), 또는 백틱(`)은 절대 포함하지 마세요.
            요일이나 제목은 단순히 줄바꿈으로 구분하세요.
            월요일 위쪽에 한칸 간격을 줘
            """.formatted(originalPlan, userRequest);

        Map<String, Object> requestBody = Map.of(
                "model", "gpt-4o-mini",
                "messages", List.of(
                        Map.of("role", "system", "content", "당신은 계획을 수정하는 AI 비서입니다."),
                        Map.of("role", "user", "content", prompt)
                )
        );

        try {
            Map<String, Object> response = webClient.post()
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
                    .bodyValue(requestBody)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();

            if (response == null || !response.containsKey("choices"))
                return "⚠️ OpenAI 응답이 비어 있습니다.";

            List<Map<String, Object>> choices = (List<Map<String, Object>>) response.get("choices");
            if (choices.isEmpty()) return "⚠️ choices 항목이 없습니다.";

            Map<String, Object> first = choices.get(0);
            Map<String, Object> message = (Map<String, Object>) first.get("message");

            if (message == null || !message.containsKey("content"))
                return "⚠️ message 내용이 없습니다.";

            return message.get("content").toString().replace("\n", "<br>");
        } catch (Exception e) {
            e.printStackTrace();
            return "⚠️ 계획을 수정하는 중 오류가 발생했습니다.";
        }
    }
}





