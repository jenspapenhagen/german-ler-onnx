package com.example.ner;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Legal entity types for German legal NER.
 */
public enum LegalEntityType {

    GS("Gesetz", "Law / Statute"),
    RS("Rechtsprechung", "Court decision"),
    GRT("Gericht", "Court"),
    LIT("Literatur", "Legal literature"),
    VT("Vertrag", "Contract / Treaty"),
    INN("Institution", "Institution"),
    PER("Person", "Person"),
    RR("Richter", "Judge"),
    EUN("EU-Norm", "EU legal norm"),
    LD("Land", "Country / State"),
    ORG("Organisation", "Organization"),
    UN("Unternehmen", "Company"),
    VO("Verordnung", "Ordinance"),
    ST("Stadt", "City"),
    VS("Vorschrift", "Regulation"),
    MRK("Marke", "Brand"),
    LDS("Landschaft", "Landscape / Region"),
    STR("Straße", "Street"),
    AN("Anwalt", "Lawyer"),

    CLS("cls_token", "cls_token"),
    PAD("pad_token", "pad_token"),
    SEP("sep_token", "sep_token"),
    MASK("mask_token", "mask_token"),
    UNK("unk_token", "unk_token");

    private final String german;
    private final String english;

    LegalEntityType(String german, String english) {
        this.german = german;
        this.english = english;
    }

    private static final Map<String, LegalEntityType> LOOKUP =
            Arrays.stream(values())
                    .collect(Collectors.toMap(Enum::name, e -> e));

    /**
     * Returns the LegalEntityType for a given code.
     *
     * @param code the entity type code (e.g., "GS", "RS")
     * @return the corresponding LegalEntityType, or UNK if not found
     */
    public static LegalEntityType fromCode(String code) {
        return LOOKUP.get(code) != null ? LOOKUP.get(code) : LegalEntityType.UNK;
    }

    /**
     * Returns all control tokens used by the tokenizer.
     *
     * @return list of control token types (CLS, PAD, SEP)
     */
    public static List<LegalEntityType> controlToken() {
        return List.of(CLS, PAD, SEP);
    }

    /**
     * Checks if the given token is a control token.
     *
     * @param token the token to check
     * @return true if control token, false otherwise
     */
    public static boolean isControlToken(String token) {
        return token.equals("[CLS]") || token.equals("[SEP]") || token.equals("[PAD]") || token.equals("[UNK]");
    }

    /**
     * @return the German label
     */
    public String getGerman() {
        return german;
    }

    /**
     * @return the English label
     */
    public String getEnglish() {
        return english;
    }
}
