package com.example.interest.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import java.util.Map;

@Data
@Builder
@AllArgsConstructor
public class InterestResult {
    private Map<String, Double> scores;
    private String top1;
    private String top2;
}

