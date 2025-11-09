package com.example.interest.controller;

import com.example.interest.service.InterestService;
import lombok.RequiredArgsConstructor;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.*;

@Controller
@RequiredArgsConstructor
public class InterestController {

    private final InterestService service;


    @GetMapping("/")
    public String home() {
        return "index";  // templates/index.html
    }

    @GetMapping("/chat")
    public String chat() {
        return "chat";   // templates/chat.html
    }

    @GetMapping("/question")
    public String redirectQuestion() {
        return "redirect:/chat";
    }


    // 🔥 Flask와 통신하는 API 추가
    @PostMapping("/predict")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> analyze(@RequestBody Map<String, String> payload) {
        try {
            String userInput = payload.get("sentence");
            if (userInput == null || userInput.isBlank()) {
                return ResponseEntity.badRequest().body(Map.of("error", "입력 문장이 비어 있습니다."));
            }

            Map<String, Double> scores = service.analyzeWithPython(userInput);
            return ResponseEntity.ok(Map.of("scores", scores));

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.internalServerError()
                    .body(Map.of("error", "Flask 서버 호출 실패", "message", e.getMessage()));
        }
    }
}





