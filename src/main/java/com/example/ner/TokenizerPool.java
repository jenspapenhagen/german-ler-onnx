package com.example.ner;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public class TokenizerPool {

    private final ThreadLocal<HuggingFaceTokenizer> local;

    public TokenizerPool() {

        this.local = ThreadLocal.withInitial(() -> {
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
        });
    }

    public HuggingFaceTokenizer get() {
        return local.get();
    }
}
