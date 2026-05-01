
package com.example.ner;

import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {

        try (GermanLerNer ner = new GermanLerNer()) {
            final List<GermanLerNer.Entity> entities =
                    ner.extractEntities("Der BGH entschied über § 280 BGB im Fall Müller." +
                    "Dies ist nach § 242 (1) BGB (abgekürzte Variante) verboten.");

            entities.stream()
                    .map(e -> e.type() + " -> " + e.text())
                    .forEach(System.out::println);
        }
    }
}
