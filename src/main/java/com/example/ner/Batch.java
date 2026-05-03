package com.example.ner;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class Batch {

    private final long[][] inputIds;
    private final long[][] attentionMask;
    private final long[][] tokenTypeIds;
    private final int batchSize;
    private final int seqLen;
    private final int[] originalLengths;
    private final int[] originalIndices;

    public Batch(long[][] inputIds, long[][] attentionMask, long[][] tokenTypeIds, int[] originalLengths, int[] originalIndices) {
        this.inputIds = inputIds;
        this.attentionMask = attentionMask;
        this.tokenTypeIds = tokenTypeIds;
        this.batchSize = inputIds.length;
        this.seqLen = inputIds[0].length;
        this.originalLengths = originalLengths;
        this.originalIndices = originalIndices;
    }

    public long[][] inputIds() {
        return inputIds;
    }

    public long[][] attentionMask() {
        return attentionMask;
    }

    public long[][] tokenTypeIds() {
        return tokenTypeIds;
    }

    public int batchSize() {
        return batchSize;
    }

    public int seqLen() {
        return seqLen;
    }

    public int[] originalLengths() {
        return originalLengths;
    }

    public int[] originalIndices() {
        return originalIndices;
    }

    public static long[] flatten(long[][] arr) {
        int rows = arr.length;
        int cols = arr[0].length;
        long[] flat = new long[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(arr[i], 0, flat, i * cols, cols);
        }
        return flat;
    }

    public static class BatchBuilder {

        private static final int MAX_BATCH_SIZE = 32;
        private static final int LENGTH_TOLERANCE = 20;

        private final List<long[]> inputIdList = new ArrayList<>();
        private final List<long[]> maskList = new ArrayList<>();
        private final List<long[]> tokenTypeList = new ArrayList<>();
        private final List<Integer> lengthList = new ArrayList<>();

        public BatchBuilder add(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) {
            inputIdList.add(inputIds.clone());
            maskList.add(attentionMask.clone());
            tokenTypeList.add(tokenTypeIds.clone());
            lengthList.add(inputIds.length);
            return this;
        }

        public List<Batch> build() {
            if (inputIdList.isEmpty()) {
                return List.of();
            }

            Integer[] indices = new Integer[lengthList.size()];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i;
            }
            Arrays.sort(indices, (a, b) -> Integer.compare(lengthList.get(a), lengthList.get(b)));

            List<Batch> batches = new ArrayList<>();
            int i = 0;
            while (i < indices.length) {
                int startLen = lengthList.get(indices[i]);
                int groupEnd = findGroupEnd(i, indices, startLen);
                int groupSize = Math.min(groupEnd - i, MAX_BATCH_SIZE);
                batches.add(createBatch(indices, i, i + groupSize));
                i += groupSize;
            }

            return batches;
        }

        private int findGroupEnd(int start, Integer[] indices, int startLen) {
            int end = start + 1;
            while (end < indices.length && lengthList.get(indices[end]) <= startLen + LENGTH_TOLERANCE) {
                end++;
            }
            return end;
        }

        private Batch createBatch(Integer[] indices, int from, int to) {
            int maxLen = 0;
            for (int idx = from; idx < to; idx++) {
                maxLen = Math.max(maxLen, inputIdList.get(indices[idx]).length);
            }

            int size = to - from;
            long[][] inputIds = new long[size][maxLen];
            long[][] attentionMask = new long[size][maxLen];
            long[][] tokenTypeIds = new long[size][maxLen];
            int[] originalLengths = new int[size];
            int[] originalIndices = new int[size];

            for (int b = 0; b < size; b++) {
                int srcIdx = indices[from + b];
                int len = inputIdList.get(srcIdx).length;
                originalLengths[b] = len;
                originalIndices[b] = srcIdx;
                System.arraycopy(inputIdList.get(srcIdx), 0, inputIds[b], 0, len);
                System.arraycopy(maskList.get(srcIdx), 0, attentionMask[b], 0, len);
                System.arraycopy(tokenTypeList.get(srcIdx), 0, tokenTypeIds[b], 0, len);
            }

            return new Batch(inputIds, attentionMask, tokenTypeIds, originalLengths, originalIndices);
        }
    }
}