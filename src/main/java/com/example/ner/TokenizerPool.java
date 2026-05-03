package com.example.ner;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Thread-localpool of HuggingFace tokenizers.
 * <p>
 * Each thread gets its own tokenizer instance.
 * </p>
 */
public class TokenizerPool {

    private final ThreadLocal<HuggingFaceTokenizer> local;

    /**
     * Creates a new TokenizerPool.
     */
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

    /**
     * Gets a tokenizer for the current thread.
     *
     * @return the tokenizer instance
     */
    public HuggingFaceTokenizer get() {
        return local.get();
    }
}
