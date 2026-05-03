package com.example.ner;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Collectors;

/**
 * German Legal Entity Recognizer using ONNX model.
 */
public class GermanLerNer implements AutoCloseable {
    private final Logger logger = LoggerFactory.getLogger(GermanLerNer.class);

    private static volatile OrtEnvironment SHARED_ENV = null;
    private static final ConcurrentMap<String, OrtSession> SESSION_CACHE = new ConcurrentHashMap<>();
    private static volatile String cachedModelPath = null;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    private final Map<Integer, String> id2label;

    private final TokenizerPool tokenizerPool;

    public record Entity(String text, LegalEntityType type) {
    }

    /**
     * Entity span with token indices (start/end).
     */
    public record EntitySpan(int start, int end, LegalEntityType type) {
    }

    /**
     * Constructs a new GermanLerNer instance.
     *
     * @throws OrtException if model loading fails
     */
    public GermanLerNer() throws OrtException {
        // Initialize shared environment once
        if (SHARED_ENV == null) {
            SHARED_ENV = OrtEnvironment.getEnvironment();
        }
        this.env = SHARED_ENV;

        // Load or reuse a cached model path
        String modelPath;
        synchronized (GermanLerNer.class) {
            modelPath = loadModelFromResources();
        }

        // Reuse a cached OrtSession per model path
        OrtSession cached = SESSION_CACHE.get(modelPath);
        if (cached == null) {
            try {
                OrtSession created = SHARED_ENV.createSession(modelPath, new OrtSession.SessionOptions());
                OrtSession prev = SESSION_CACHE.putIfAbsent(modelPath, created);
                cached = prev != null ? prev : created;
            } catch (OrtException e) {
                throw e;
            }
        }

        this.tokenizerPool = new TokenizerPool();
        this.session = cached;
        this.tokenizer = tokenizerPool.get();
        this.id2label = loadLabelsFromResources();
        logger.info("loading done");
    }

    private String loadModelFromResources() {
        long t0 = System.nanoTime();
        // Return cached path if available
        if (cachedModelPath != null && Files.exists(Path.of(cachedModelPath))) {
            final long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
            if (Boolean.getBoolean("onnxperf.logLoadTime")) {
                logger.info("[ONNX] loadModelFromResources (cached) time: " + elapsedMs + " ms");
            }
            return cachedModelPath;
        }
        try (InputStream resourceAsStream = getClass().getClassLoader().getResourceAsStream("model_int4.onnx")) {
            if (resourceAsStream == null) {
                throw new IllegalStateException("model_int4.onnx not found in resources");
            }
            final Path tempFile = Files.createTempFile("german-ler", ".onnx");
            Files.copy(resourceAsStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
            tempFile.toFile().deleteOnExit();

            cachedModelPath = tempFile.toAbsolutePath().toString();
            long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
            if (Boolean.getBoolean("onnxperf.logLoadTime")) {
                logger.info("[ONNX] loadModelFromResources load time: " + elapsedMs + " ms");
            }
            return cachedModelPath;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }


    private Map<Integer, String> loadLabelsFromResources() {
        try (InputStream resourceAsStream = getClass().getClassLoader().getResourceAsStream("config.json")) {

            if (resourceAsStream == null) {
                throw new IllegalStateException("config.json not found");
            }

            final ObjectMapper mapper = new ObjectMapper();
            final JsonNode root = mapper.readTree(resourceAsStream);

            final JsonNode labels = root.get("id2label");

            return labels.properties().stream()
                    .collect(Collectors.toMap(
                            e -> Integer.parseInt(e.getKey()),
                            e -> e.getValue().asString()
                    ));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * Extracts entity spans using char spans (returns original text directly).
     *
     * @param text the original German legal text
     * @return list of extracted entities with original text
     * @throws OrtException if inference fails
     */
    public List<Entity> extractEntities(final String text) throws OrtException {
        final Encoding encoding = tokenizer.encode(text);

        final long[] inputIds = encoding.getIds();
        final long[] attentionMask = encoding.getAttentionMask();
        final long[] tokenTypeIds = encoding.getTypeIds();

        final Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", OnnxTensor.createTensor(env, new long[][]{inputIds}));
        inputs.put("attention_mask", OnnxTensor.createTensor(env, new long[][]{attentionMask}));
        inputs.put("token_type_ids", OnnxTensor.createTensor(env, new long[][]{tokenTypeIds}));

        final OrtSession.Result result = session.run(inputs);
        final float[][][] logits = (float[][][]) result.get(0).getValue();

        final List<EntitySpan> spans = decodeSpansFromLogits(logits[0], id2label);
        final String[] tokens = encoding.getTokens();

        final List<Entity> entities = new ArrayList<>(spans.size());
        for (EntitySpan span : spans) {
            String spanText = spanToText(span, tokens);
            entities.add(new Entity(spanText, span.type()));
        }
        return entities;
    }

    /**
     * Decodes entity spans from logits using BIO tagging.
     *
     * @param logits   2D array [seq_len][num_labels]
     * @param id2label label mapping
     * @return list of entity spans
     */
    private static List<EntitySpan> decodeSpansFromLogits(final float[][] logits, final Map<Integer, String> id2label) {
        final List<EntitySpan> result = new ArrayList<>(8);

        LegalEntityType currentType = null;
        int start = -1;

        for (int i = 0; i < logits.length; i++) {
            int pred = argmax(logits[i]);
            final String label = id2label.get(pred);

            // Skip null labels
            if (label == null) {
                continue;
            }

            // O fast path
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
     * Extracts original text from span by joining tokens.
     *
     * @param span   entity span with token indices
     * @param tokens tokenizer tokens
     * @return extracted text span
     */
    private static String spanToText(final EntitySpan span, final String[] tokens) {
        final StringBuilder stringBuilder = new StringBuilder();
        for (int i = span.start(); i < span.end(); i++) {
            final String token = tokens[i];
            if (LegalEntityType.isControlToken(token)) {
                continue;
            }
            // Handle BERT subword tokens
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

    /**
     * Argmax for single row (no allocations).
     */
    private static int argmax(final float[] row) {
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

    /**
     * Closes the NER runtime and releases resources.
     *
     * @throws Exception if closing fails
     */
    public void close() throws Exception {
        session.close();
        env.close();
    }
}
