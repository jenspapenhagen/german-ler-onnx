package com.example.ner;

public record BioTag(char prefix, LegalEntityType type) {

    static BioTag parse(String label) {

        if (label == null || label.equals("O")) {
            return new BioTag('O', null);
        }

        char prefix = label.charAt(0); // B / I / maybe E/S later
        String code = label.substring(2);

        return new BioTag(prefix, LegalEntityType.fromCode(code));
    }
}
