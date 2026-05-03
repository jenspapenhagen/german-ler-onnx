package com.example.ner;

import org.junit.jupiter.api.*;
import java.util.*;

/**
 * Benchmark - runs all tests in single method to avoid ONNX session issues.
 */
class GermanLerNerBenchmark {

    private static final String[] SENTENCES = {
        "Der BGH entschied über § 280 BGB im Fall Müller.",
        "Das Gericht hat die Klage abgewiesen.",
        "Die Verordnung gilt ab dem 1. Januar 2024.",
    };

    private GermanLerNer ner;

    @BeforeEach
    void setUp() throws Exception {
        ner = new GermanLerNer();
    }

    @AfterEach
    void tearDown() throws Exception {
        if (ner != null) ner.close();
    }

    @Test
    void runAllBenchmarks() throws Exception {
        System.out.println();
        System.out.println("═══════════════════════════════════════════════════════════════");
        System.out.println("BENCHMARK RESULTS (5 runs, 3 warmup, median)");
        System.out.println("═══════════════════════════════════════════════════════════════");
        System.out.printf("%-35s | %12s%n", "Operation", "Median Time");
        System.out.println("-".repeat(50));

        // ========== WARMUP ==========
        for (int i = 0; i < 3; i++) {
            for (String s : SENTENCES) ner.extractEntities(s);
        }

        // ========== 1. BASELINE FULL ==========
        List<Long> times = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            long t0 = System.nanoTime();
            for (String s : SENTENCES) ner.extractEntities(s);
            times.add(System.nanoTime() - t0);
        }
        times.sort(Long::compareTo);
        System.out.printf("%-35s | %10.2f ms%n", "1. Baseline (full pipeline)", times.get(2) / 1e6);

        // ========== 2. EXTRACT SPANS ONLY ==========
        times.clear();
        for (int i = 0; i < 5; i++) {
            long t0 = System.nanoTime();
            for (String s : SENTENCES) ner.extractEntitySpans(s);
            times.add(System.nanoTime() - t0);
        }
        times.sort(Long::compareTo);
        System.out.printf("%-35s | %10.2f ms%n", "2. Extract spans only", times.get(2) / 1e6);

        // ========== 3. TOKENIZATION ==========
        times.clear();
        for (int i = 0; i < 5; i++) {
            long t0 = System.nanoTime();
            for (String s : SENTENCES) ner.tokenizer.encode(s);
            times.add(System.nanoTime() - t0);
        }
        times.sort(Long::compareTo);
        System.out.printf("%-35s | %10.2f ms%n", "3. Tokenization", times.get(2) / 1e6);

        // ========== 4. ONNX INFERENCE ==========
        times.clear();
        for (int i = 0; i < 5; i++) {
            long t0 = System.nanoTime();
            for (String s : SENTENCES) ner.runInference(s);
            times.add(System.nanoTime() - t0);
        }
        times.sort(Long::compareTo);
        System.out.printf("%-35s | %10.2f ms%n", "4. ONNX inference", times.get(2) / 1e6);

        // ========== 5. DECODE SPANS ==========
        float[][] logits = ner.runInference(SENTENCES[0]);
        Map<Integer, String> id2label = new HashMap<>();
        for (int i = 0; i < 20; i++) id2label.put(i, "O");

        times.clear();
        for (int i = 0; i < 5; i++) {
            long t0 = System.nanoTime();
            for (int j = 0; j < 100; j++) {
                GermanLerNer.decodeSpansFromLogits(logits, id2label);
            }
            times.add(System.nanoTime() - t0);
        }
        times.sort(Long::compareTo);
        System.out.printf("%-35s | %10.2f us%n", "5. Decode (100 iter)", times.get(2) / 1e3);

        // ========== 6. TEXT EXTRACTION ==========
        String[] tokens = ner.tokenizer.encode(SENTENCES[0]).getTokens();
        var span = new GermanLerNer.EntitySpan(1, 5, LegalEntityType.GS);

        times.clear();
        for (int i = 0; i < 5; i++) {
            long t0 = System.nanoTime();
            for (int j = 0; j < 1000; j++) {
                GermanLerNer.spanToText(span, tokens);
            }
            times.add(System.nanoTime() - t0);
        }
        times.sort(Long::compareTo);
        System.out.printf("%-35s | %10.2f us%n", "6. spanToText (1000 iter)", times.get(2) / 1e3);

        System.out.println("-".repeat(50));
    }
}