
package com.example.ner;

import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        final GermanLerModel germanLerModel = GermanLerModel.getInstance();

        final GermanLerNer ner = new GermanLerNer(germanLerModel);

        final List<String> sentences = List.of(
                "Der BGH entschied über § 280 BGB im Fall Müller. Dies ist nach § 242 (1) BGB (abgekürzte Variante) verboten.",
                "Das Landgericht Berlin hat im Verfahren XYZ entschieden.",
                "Nach § 242 BGB ist dies verboten.",
                "Die EU-Norm 2019/1024 regelt diesen Bereich.",
                "Das Unternehmen Siemens AG mit Sitz in München."
        );

        System.out.println("=== Batch Processing with Dynamic Batching ===");
        System.out.println("Input: " + sentences.size() + " sentences\n");

        long startTime = System.nanoTime();
        List<List<GermanLerNer.Entity>> batchResults = ner.extractEntitiesBatch(sentences);
        long batchTime = System.nanoTime() - startTime;

        for (int i = 0; i < sentences.size(); i++) {
            System.out.println("Sentence " + (i + 1) + ": " + sentences.get(i));
            List<GermanLerNer.Entity> entities = batchResults.get(i);
            if (entities.isEmpty()) {
                System.out.println("  -> No entities found");
            } else {
                for (GermanLerNer.Entity e : entities) {
                    System.out.println("  -> " + e.text() + " (" + e.type().getGerman() + ")");
                }
            }
            System.out.println();
        }

        System.out.println("Batch processing time: " + (batchTime / 1_000_000) + "ms");

        System.out.println("\n=== Single Processing (for comparison) ===");
        for (String sentence : sentences) {
            long singleStart = System.nanoTime();
            List<GermanLerNer.Entity> singleEntities = ner.extractEntities(sentence);
            long singleTime = System.nanoTime() - singleStart;
            System.out.println("Time: " + (singleTime / 1_000_000) + "ms -> " + singleEntities.size() + " entities");
        }
    }
}
