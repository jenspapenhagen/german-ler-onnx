
package com.example.ner;

import java.util.List;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) throws Exception {

        try (GermanLerNer ner = new GermanLerNer()) {
            final String sentence = "Der BGH entschied über § 280 BGB im Fall Müller. Dies ist nach § 242 (1) BGB (abgekürzte Variante) verboten.";

            System.out.println("Orginal Sentence: " + sentence);

            final List<GermanLerNer.Entity> entities = ner.extractEntities(sentence);

            final String collect = entities.stream()
                    .map(e -> e.text() + " (" + e.type().getGerman() + ")")
                    .collect(Collectors.joining(" "));

            System.out.println("Extracted Keywords: " + collect);
        }
    }
}
