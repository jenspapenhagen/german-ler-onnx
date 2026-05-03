package com.example.ner;

import org.junit.jupiter.api.*;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for GermanLerNer - decode and spanToText.
 */
class GermanLerNerTest {

    @Test
    @DisplayName("decodeSpans - basic entity")
    void testDecodeBasic() {
        float[][] logits = {{0.9f, 0.1f}, {0.1f, 0.9f}, {0.9f, 0.1f}};
        Map<Integer, String> id2label = Map.of(0, "O", 1, "B-GS");

        List<GermanLerNer.EntitySpan> spans = GermanLerNer.decodeSpansFromLogits(logits, id2label);

        assertEquals(1, spans.size());
        assertEquals(1, spans.getFirst().start());
        assertEquals(2, spans.getFirst().end());
        assertEquals(LegalEntityType.GS, spans.getFirst().type());
    }

    @Test
    @DisplayName("decodeSpans - multi-token")
    void testDecodeMulti() {
        float[][] logits = {{0.9f, 0f}, {0f, 0.9f}, {0f, 0.8f}, {0.9f, 0f}};
        Map<Integer, String> id2label = Map.of(0, "O", 1, "B-GS", 2, "I-GS");

        List<GermanLerNer.EntitySpan> spans = GermanLerNer.decodeSpansFromLogits(logits, id2label);

        assertEquals(1, spans.size());
        assertEquals(1, spans.getFirst().start());
        assertEquals(3, spans.getFirst().end());
    }

    @Test
    @DisplayName("spanToText - subword merge")
    void testSpanToText() {
        String[] tokens = {"Der", "BGH", "##entschied"};
        var span = new GermanLerNer.EntitySpan(1, 3, LegalEntityType.GRT);

        String text = GermanLerNer.spanToText(span, tokens);
        assertEquals("BGH entschied", text);
    }

    @Test
    @DisplayName("spanToText - control tokens")
    void testSkipControl() {
        String[] tokens = {"[CLS]", "Der", "BGH", "[SEP]"};
        var span = new GermanLerNer.EntitySpan(0, 4, LegalEntityType.GRT);

        String text = GermanLerNer.spanToText(span, tokens);
        assertEquals("Der BGH", text);
    }

    @Test
    @DisplayName("null label returns empty")
    void testNullLabel() {
        float[][] logits = {{0.5f}};
        List<GermanLerNer.EntitySpan> spans = GermanLerNer.decodeSpansFromLogits(logits, new HashMap<>());
        assertTrue(spans.isEmpty());
    }

    @Test
    @DisplayName("UNK type ignored")
    void testUnk() {
        float[][] logits = {{0f, 0.9f}};
        List<GermanLerNer.EntitySpan> spans = GermanLerNer.decodeSpansFromLogits(logits, Map.of(0, "O", 1, "B-XXX"));
        assertTrue(spans.isEmpty());
    }
}