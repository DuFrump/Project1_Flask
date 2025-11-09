package com.example.interest.controller;


import com.example.interest.service.OpenAIService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class ChatController {

    private final OpenAIService service;

    @PostMapping("/ask")
    public Map<String, Object> ask(@RequestBody Map<String, String> req) {
        String userMessage = req.get("message");
        return service.getResponse(userMessage);
    }
}

