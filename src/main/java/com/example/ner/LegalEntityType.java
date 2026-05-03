package com.example.ner;

import java.util.List;

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

    public static List<LegalEntityType> controllToken(){
        return List.of(CLS, PAD, SEP);
    }

    public String getGerman() {
        return german;
    }

    public String getEnglish() {
        return english;
    }
}
