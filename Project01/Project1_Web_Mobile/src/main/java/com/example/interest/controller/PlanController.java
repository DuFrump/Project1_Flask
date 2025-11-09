package com.example.interest.controller;

import com.example.interest.service.PlanService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Map;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/plan")
public class PlanController {

    private final PlanService planService;

    @PostMapping
    public Map<String, Object> getPlan(@RequestBody Map<String, Object> body) {
        String topic = (String) body.get("topic");
        List<String> answers = (List<String>) body.get("answers");
        String plan = planService.generatePlan(topic, answers);
        return Map.of("plan", plan);
    }

    @PostMapping("/refine")
    public Map<String, Object> refinePlan(@RequestBody Map<String, String> body) {
        String originalPlan = body.get("plan");
        String userRequest = body.get("request");
        String refined = planService.refinePlan(originalPlan, userRequest);
        return Map.of("plan", refined);
    }
}





