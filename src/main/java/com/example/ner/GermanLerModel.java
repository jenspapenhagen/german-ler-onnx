package com.example.ner;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * ONNX model wrapper for German LER.
 * <p>
 * Handles model loading, caching, and inference.
 * Thread-safe singleton - loads once, reuses everywhere.
 * </p>
 */
public class GermanLerModel implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(GermanLerModel.class);

    private static volatile GermanLerModel INSTANCE = null;
    private static volatile String CACHED_MODEL_PATH = null;

    private static final ConcurrentMap<String, OrtSession> SESSION_CACHE = new ConcurrentHashMap<>();

    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    private final Map<Integer, String> id2label;

    private GermanLerModel() throws OrtException {
        // Single environment
        this.env = OrtEnvironment.getEnvironment();

        // Load model
        this.session = createSession();

        // Init tokenizer
        this.tokenizer = createTokenizer();

        // Load labels
        this.id2label = loadLabels();

        LOG.info("GermanLerModel loaded");
    }

    /**
     * Gets the singleton instance.
     */
    public static GermanLerModel getInstance() throws OrtException {
        if (INSTANCE == null) {
            synchronized (GermanLerModel.class) {
                if (INSTANCE == null) {
                    INSTANCE = new GermanLerModel();
                }
            }
        }
        return INSTANCE;
    }

    /**
     * Gets the ONNX session.
     */
    public OrtSession session() {
        return session;
    }

    /**
     * Gets the tokenizer.
     */
    public HuggingFaceTokenizer tokenizer() {
        return tokenizer;
    }

    /**
     * Gets label mapping.
     */
    public Map<Integer, String> id2label() {
        return id2label;
    }

    /**
     * Runs inference, returns logits.
     */
    public float[][] runInference(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) throws OrtException {
        Map<String, OnnxTensor> inputs = INPUT_MAP.get();
        inputs.clear();
        inputs.put("input_ids", OnnxTensor.createTensor(env, new long[][]{inputIds}));
        inputs.put("attention_mask", OnnxTensor.createTensor(env, new long[][]{attentionMask}));
        inputs.put("token_type_ids", OnnxTensor.createTensor(env, new long[][]{tokenTypeIds}));

        OrtSession.Result result = session.run(inputs);
        float[][][] logits = (float[][][]) result.get(0).getValue();
        return logits[0];
    }

    /**
     * Thread-local input map
     */
    private static final ThreadLocal<Map<String, OnnxTensor>> INPUT_MAP = ThreadLocal.withInitial(ConcurrentHashMap::new);

    private OrtSession createSession() throws OrtException {
        final String modelPath = loadModelPath();
        OrtSession cached = SESSION_CACHE.get(modelPath);
        if (cached == null) {
            OrtSession created = env.createSession(modelPath, new OrtSession.SessionOptions());
            OrtSession prev = SESSION_CACHE.putIfAbsent(modelPath, created);
            cached = prev != null ? prev : created;
        }
        return cached;
    }

    private String loadModelPath() throws IllegalStateException {
        if (CACHED_MODEL_PATH != null && Files.exists(Path.of(CACHED_MODEL_PATH))) {
            return CACHED_MODEL_PATH;
        }
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("model_int4.onnx")) {
            if (is == null) {
                throw new IllegalStateException("model_int4.onnx not found");
            }
            final Path temp = Files.createTempFile("german-ler", ".onnx");
            Files.copy(is, temp, StandardCopyOption.REPLACE_EXISTING);
            temp.toFile().deleteOnExit();
            CACHED_MODEL_PATH = temp.toAbsolutePath().toString();
            return CACHED_MODEL_PATH;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private HuggingFaceTokenizer createTokenizer() throws IllegalStateException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("tokenizer.json")) {
            if (is == null) {
                throw new IllegalStateException("tokenizer.json not found");
            }
            final Path temp = Files.createTempFile("tokenizer", ".json");
            Files.copy(is, temp, StandardCopyOption.REPLACE_EXISTING);
            temp.toFile().deleteOnExit();
            return HuggingFaceTokenizer.newInstance(temp.toAbsolutePath());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private Map<Integer, String> loadLabels() throws IllegalStateException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("config.json")) {
            if (is == null) {
                throw new IllegalStateException("config.json not found");
            }
            final ObjectMapper mapper = new ObjectMapper();
            final JsonNode root = mapper.readTree(is);
            final JsonNode labels = root.get("id2label");
            return labels.properties().stream()
                    .collect(java.util.stream.Collectors.toMap(
                            e -> Integer.parseInt(e.getKey()),
                            e -> e.getValue().asString()
                    ));
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void close() throws Exception {
        // Don't really close - singleton
    }
}