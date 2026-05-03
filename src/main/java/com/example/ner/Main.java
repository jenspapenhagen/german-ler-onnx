
package com.example.ner;

import java.util.List;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) throws Exception {
        final GermanLerModel germanLerModel = GermanLerModel.getInstance();

        final GermanLerNer ner = new GermanLerNer(germanLerModel);

        final String sentence = "Der BGH entschied über § 280 BGB im Fall Müller. Dies ist nach § 242 (1) BGB (abgekürzte Variante) verboten.";

        System.out.println("Orginal Sentence: " + sentence);

        final List<GermanLerNer.Entity> entities = ner.extractEntities(sentence);

        final String collect = entities.stream()
                .map(e -> e.text() + " (" + e.type().getGerman() + ")")
                .collect(Collectors.joining(" \n"));

        System.out.println("Extracted Keywords: \n" + collect);
    }
}
