
package com.example.ner;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
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

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.util.stream.Collectors;

public class GermanLerNer implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    private final Map<Integer, String> id2label;

    public record Entity(String type, String text) {
    }

    public GermanLerNer() throws OrtException {
        this.env = OrtEnvironment.getEnvironment();

        this.session = env.createSession(loadModelFromResources(), new OrtSession.SessionOptions());
        this.tokenizer = loadTokenizer();
        this.id2label = loadLabelsFromResources();
    }

    private String loadModelFromResources() {
        try (InputStream resourceAsStream = getClass().getClassLoader().getResourceAsStream("model_int4.onnx")) {
            if (resourceAsStream == null) {
                throw new IllegalStateException("model_int4.onnx not found in resources");
            }
            final Path tempFile = Files.createTempFile("german-ler", ".onnx");
            Files.copy(resourceAsStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
            tempFile.toFile().deleteOnExit();

            return tempFile.toAbsolutePath().toString();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }


    private HuggingFaceTokenizer loadTokenizer() {
        try (InputStream resourceAsStream = getClass().getClassLoader().getResourceAsStream("tokenizer.json")) {
            if (resourceAsStream == null) {
                throw new IllegalStateException("tokenizer.json not found in resources");
            }

            final Path tokenizerPath = Files.createTempFile("tokenizer", ".json");
            Files.copy(resourceAsStream, tokenizerPath, StandardCopyOption.REPLACE_EXISTING);
            tokenizerPath.toFile().deleteOnExit();

            return HuggingFaceTokenizer.newInstance(tokenizerPath.toAbsolutePath());
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
        final int[] predictions = argmax(logits[0]);

        final List<String> tokens = List.of(encoding.getTokens());

        final List<Entity> entities = new ArrayList<>();

        String currentType = null;
        List<String> currentTokens = new ArrayList<>();

        for (int i = 0; i < predictions.length; i++) {
            final String token = tokens.get(i);
            //skip token
//            "cls_token": "[CLS]",
//            "mask_token": "[MASK]",
//            "pad_token": "[PAD]",
//            "sep_token": "[SEP]",
//            "unk_token": "[UNK]"
            if (token.equals("[CLS]") || token.equals("[SEP]") || token.equals("[PAD]")) {
                continue;
            }

            final String label = id2label.get(predictions[i]);

            if (label == null || label.equals("O")) {
                if (currentType != null) {
                    entities.add(new Entity(currentType, String.join(" ", currentTokens)));
                    currentTokens.clear();
                    currentType = null;
                }
                continue;
            }

            if (label.startsWith("B-")) {
                if (currentType != null) {
                    entities.add(new Entity(currentType, String.join(" ", currentTokens)));
                }

                currentType = label.substring(2);
                currentTokens = new ArrayList<>();
                currentTokens.add(token);
            } else if (label.startsWith("I-") && label.substring(2).equals(currentType)) {
                currentTokens.add(token);
            } else {
                if (currentType != null) {
                    entities.add(new Entity(currentType, String.join(" ", currentTokens)));
                }
                currentType = null;
                currentTokens.clear();
            }
        }

        if (currentType != null) {
            entities.add(new Entity(currentType, String.join(" ", currentTokens)));
        }

        return entities;
    }

    private int[] argmax(final float[][] logits) {
        final int[] res = new int[logits.length];

        for (int i = 0; i < logits.length; i++) {
            float max = Float.NEGATIVE_INFINITY;
            int idx = 0;

            for (int j = 0; j < logits[i].length; j++) {
                if (logits[i][j] > max) {
                    max = logits[i][j];
                    idx = j;
                }
            }
            res[i] = idx;
        }
        return res;
    }

    public void close() throws Exception {
        session.close();
        env.close();
    }
}
