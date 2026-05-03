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
        float[][] logits = {{0.9f, 0f, 0f}, {0f, 0.9f, 0f}, {0f, 0.1f, 0.8f}, {0.9f, 0f, 0f}};
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

    @Test
    @DisplayName("Batch - single sequence")
    void testBatchSingle() {
        long[][] inputIds = {{101, 202, 203, 102}};
        long[][] attentionMask = {{1, 1, 1, 1}};
        long[][] tokenTypeIds = {{0, 0, 0, 0}};
        int[] lengths = {4};
        int[] indices = {0};

        Batch batch = new Batch(inputIds, attentionMask, tokenTypeIds, lengths, indices);

        assertEquals(1, batch.batchSize());
        assertEquals(4, batch.seqLen());
        assertEquals(4, batch.originalLengths()[0]);
    }

    @Test
    @DisplayName("Batch - multiple sequences padded correctly")
    void testBatchPadding() {
        long[][] inputIds = {
            {101, 202, 203, 102, 0, 0},
            {101, 204, 205, 206, 207, 102}
        };
        long[][] attentionMask = {
            {1, 1, 1, 1, 0, 0},
            {1, 1, 1, 1, 1, 1}
        };
        long[][] tokenTypeIds = {
            {0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0}
        };
        int[] lengths = {4, 6};
        int[] indices = {0, 1};

        Batch batch = new Batch(inputIds, attentionMask, tokenTypeIds, lengths, indices);

        assertEquals(2, batch.batchSize());
        assertEquals(6, batch.seqLen());
        assertArrayEquals(new int[]{4, 6}, batch.originalLengths());
    }

    @Test
    @DisplayName("BatchBuilder - groups similar length")
    void testBatchBuilderGroupsSimilarLength() {
        Batch.BatchBuilder builder = new Batch.BatchBuilder();

        builder.add(new long[]{101, 102, 103}, new long[]{1, 1, 1}, new long[]{0, 0, 0});
        builder.add(new long[]{101, 102}, new long[]{1, 1}, new long[]{0, 0});
        builder.add(new long[]{101, 102, 103, 104, 105}, new long[]{1, 1, 1, 1, 1}, new long[]{0, 0, 0, 0, 0});
        builder.add(new long[]{101, 102, 103, 104}, new long[]{1, 1, 1, 1}, new long[]{0, 0, 0, 0});

        List<Batch> batches = builder.build();

        assertTrue(batches.size() >= 1);
    }

    @Test
    @DisplayName("BatchBuilder - empty input returns empty list")
    void testBatchBuilderEmpty() {
        Batch.BatchBuilder builder = new Batch.BatchBuilder();
        assertTrue(builder.build().isEmpty());
    }

    @Test
    @DisplayName("Batch flatten works correctly")
    void testBatchFlatten() {
        long[][] inputIds = {
            {1, 2, 3},
            {4, 5, 6}
        };

        long[] flat = Batch.flatten(inputIds);

        assertEquals(6, flat.length);
        assertArrayEquals(new long[]{1, 2, 3, 4, 5, 6}, flat);
    }
}