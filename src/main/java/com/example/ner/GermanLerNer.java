package com.example.ner;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OrtException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
                if (currentType != null && currentType != type) {
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
            if (!stringBuilder.isEmpty()) {
                stringBuilder.append(" ");
            }
            if (token.startsWith("##")) {
                stringBuilder.append(token.substring(2));
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

    public List<List<Entity>> extractEntitiesBatch(List<String> texts) throws OrtException {
        if (texts.isEmpty()) {
            return List.of();
        }

        String[] textArray = texts.toArray(new String[0]);
        final var tokenizer = model.tokenizer();
        final var builder = new Batch.BatchBuilder();

        for (String text : texts) {
            Encoding encoding = tokenizer.encode(text);
            builder.add(encoding.getIds(), encoding.getAttentionMask(), encoding.getTypeIds());
        }

        List<Batch> batches = builder.build();
        List<List<Entity>> orderedResults = new ArrayList<>(texts.size());

        for (Batch batch : batches) {
            float[][][] logits = model.runBatchInference(batch);
            int[] originalIndices = batch.originalIndices();
            String[] batchTexts = new String[batch.batchSize()];
            for (int i = 0; i < batch.batchSize(); i++) {
                batchTexts[i] = textArray[originalIndices[i]];
            }
            List<List<Entity>> batchResults = decodeBatchParallel(logits, batch.originalLengths(), batchTexts);
            for (int i = 0; i < batch.batchSize(); i++) {
                int origIdx = originalIndices[i];
                while (orderedResults.size() <= origIdx) {
                    orderedResults.add(null);
                }
                orderedResults.set(origIdx, batchResults.get(i));
            }
        }

        return orderedResults;
    }

    private List<List<Entity>> decodeBatchParallel(final float[][][] logits, final int[] originalLengths, final String[] originalTexts) {
        final int batchSize = logits.length;
        final List<Future<List<Entity>>> futures = new ArrayList<>();

        List<List<Entity>> results;
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int b = 0; b < batchSize; b++) {
                final int idx = b;
                futures.add(executor.submit(() -> decodeSingle(idx, logits, originalLengths, originalTexts[idx])));
            }

            results = new ArrayList<>(batchSize);
            for (Future<List<Entity>> f : futures) {
                try {
                    results.add(f.get());
                } catch (Exception e) {
                    results.add(List.of());
                }
            }
            executor.shutdown();
        }

        return results;
    }

    private List<Entity> decodeSingle(final int idx, final float[][][] logits, final int[] originalLengths, final String originalText) {
        final float[][] seqLogits = Arrays.copyOfRange(logits[idx], 0, originalLengths[idx]);
        final List<EntitySpan> spans = decodeSpansFromLogits(seqLogits, model.id2label());
        final String[] tokens = model.tokenizer().encode(originalText).getTokens();

        final List<Entity> entities = new ArrayList<>(spans.size());
        for (final EntitySpan span : spans) {
            entities.add(new Entity(spanToText(span, tokens), span.type()));
        }
        return entities;
    }
}