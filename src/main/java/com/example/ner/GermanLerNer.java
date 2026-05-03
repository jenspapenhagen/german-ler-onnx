package com.example.ner;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OrtException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * German Legal Entity Recognizer.
 * <p>
 * Lightweight wrapper - uses {@link GermanLerModel} for heavy lifting.
 * Can be instantiated per-thread with minimal overhead.
 * </p>
 */
public class GermanLerNer {

    private final GermanLerModel model;

    public record Entity(String text, LegalEntityType type) {
    }

    public record EntitySpan(int start, int end, LegalEntityType type) {
    }

    /**
     * Creates NER with given model.
     */
    public GermanLerNer(GermanLerModel model) {
        this.model = model;
    }

    /**
     * Extracts entities with original text.
     */
    public List<Entity> extractEntities(final String text) throws OrtException {
        final List<EntitySpan> spans = extractEntitySpans(text);
        final String[] tokens = model.tokenizer().encode(text).getTokens();

        final List<Entity> entities = new ArrayList<>(spans.size());
        for (final EntitySpan span : spans) {
            final String spanText = spanToText(span, tokens);
            entities.add(new Entity(spanText, span.type()));
        }
        return entities;
    }

    /**
     * Extracts entity spans (token indices only).
     */
    public List<EntitySpan> extractEntitySpans(final String text) throws OrtException {
        final HuggingFaceTokenizer tokenizer = model.tokenizer();
        final Encoding encoding = tokenizer.encode(text);

        final long[] ids = encoding.getIds();
        final long[] mask = encoding.getAttentionMask();
        final long[] typeIds = encoding.getTypeIds();

        final float[][] logits = model.runInference(ids, mask, typeIds);
        return decodeSpansFromLogits(logits, model.id2label());
    }

    /**
     * Decode spans from logits.
     */
    static List<EntitySpan> decodeSpansFromLogits(final float[][] logits, final Map<Integer, String> id2label) {
        final List<EntitySpan> result = new ArrayList<>(8);
        LegalEntityType currentType = null;
        int start = -1;

        for (int i = 0; i < logits.length; i++) {
            final int pred = argmax(logits[i]);
            final String label = id2label.get(pred);

            if (label == null) {
                continue;
            }

            if (label.charAt(0) == 'O') {
                if (currentType != null) {
                    result.add(new EntitySpan(start, i, currentType));
                    currentType = null;
                }
                continue;
            }

            final char prefix = label.charAt(0);
            final String code = label.substring(2);
            final LegalEntityType type = LegalEntityType.fromCode(code);

            if (type == LegalEntityType.UNK) {
                if (currentType != null) {
                    result.add(new EntitySpan(start, i, currentType));
                    currentType = null;
                }
                continue;
            }

            if (prefix == 'B') {
                if (currentType != null) {
                    result.add(new EntitySpan(start, i, currentType));
                }
                currentType = type;
                start = i;
            } else if (prefix == 'I') {
                if (currentType != type) {
                    if (currentType != null) {
                        result.add(new EntitySpan(start, i, currentType));
                    }
                    currentType = null;
                }
            } else {
                if (currentType != null) {
                    result.add(new EntitySpan(start, i, currentType));
                    currentType = null;
                }
            }
        }

        if (currentType != null) {
            result.add(new EntitySpan(start, logits.length, currentType));
        }
        return result;
    }

    /**
     * Extract text from span using tokens.
     */
    static String spanToText(final EntitySpan span, final String[] tokens) {
        final StringBuilder stringBuilder = new StringBuilder();
        for (int i = span.start(); i < span.end(); i++) {
            final String token = tokens[i];
            if (LegalEntityType.isControlToken(token)) {
                continue;
            }
            if (token.startsWith("##")) {
                stringBuilder.append(token.substring(2));
            } else if (!stringBuilder.isEmpty()) {
                stringBuilder.append(" ").append(token);
            } else {
                stringBuilder.append(token);
            }
        }
        return stringBuilder.toString().trim();
    }

    private static int argmax(float[] row) {
        int idx = 0;
        float max = row[0];
        for (int i = 1; i < row.length; i++) {
            float v = row[i];
            if (v > max) {
                max = v;
                idx = i;
            }
        }
        return idx;
    }
}