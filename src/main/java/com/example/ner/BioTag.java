package com.example.ner;

/**
 * BIO tag for NER parsing.
 * <p>
 * B = Begin, I = Inside, O = Outside.
 * </p>
 */
public record BioTag(char prefix, LegalEntityType type) {

    /**
     * Parses a label string into a BioTag.
     *
     * @param label the label (e.g., "B-GS", "O", "I-RS")
     * @return the parsed BioTag
     */
    static BioTag parse(String label) {

        if (label == null || label.equals("O")) {
            return new BioTag('O', null);
        }

        char prefix = label.charAt(0);
        String code = label.substring(2);

        return new BioTag(prefix, LegalEntityType.fromCode(code));
    }
}
