# German LER ONNX Java Library

German Legal Entity Recognition (LER) using BERT models in ONNX format for efficient inference on the JVM.

## Model

This library uses the `mayflowergmbh/bert-german-ler-onnx-int4` 
model from HuggingFace, 
converted to ONNX format with INT4 quantization for optimized inference. [Link](https://huggingface.co/mayflowergmbh/bert-german-ler-onnx-int4)

For details on the underlying model architecture and training, see the original paper: https://arxiv.org/abs/2003.13016

## Features

- **ONNX Runtime Inference**: Fast native inference on JVM via ONNX Runtime
- **Dynamic Batching**: Groups inputs of similar length to minimize padding waste
- **Virtual Threads**: Uses Java 21+ virtual threads for parallel decoding
- **Thread-Safe Singleton**: Model loaded once, reused across threads

## Requirements

- Java 21 or higher
- Maven 3.6+

## Quick Start

```java
import com.example.ner.*;

// Get model instance (singleton)
GermanLerModel model = GermanLerModel.getInstance();

// Create NER handler
GermanLerNer ner = new GermanLerNer(model);

// Extract entities from single text
List<GermanLerNer.Entity> entities = ner.extractEntities("Der BGH entschied über § 280 BGB.");

// Or batch multiple texts
List<String> texts = List.of(
    "Der BGH entschied...",
    "Das Landgericht..."
);
List<List<GermanLerNer.Entity>> batchResults = ner.extractEntitiesBatch(texts);
```

## Entity Types

The model recognizes 19 legal entity types:

| Code | German | English | Dataset Share |
|------|--------|---------|---------------|
| GS | Gesetz | Law / Statute | 34.53% |
| RS | Rechtsprechung | Court decision | 23.46% |
| GRT | Gericht | Court | 5.99% |
| LIT | Literatur | Legal literature | 5.60% |
| VT | Vertrag | Contract / Treaty | 5.34% |
| INN | Institution | Institution | 4.09% |
| PER | Person | Person | 3.26% |
| RR | Richter | Judge | 2.83% |
| EUN | EU-Norm | EU legal norm | 2.79% |
| LD | Land | Country / State | 2.66% |
| ORG | Organisation | Organization | 2.17% |
| UN | Unternehmen | Company | 1.97% |
| VO | Verordnung | Ordinance | 1.49% |
| ST | Stadt | City | 1.31% |
| VS | Vorschrift | Regulation | 1.13% |
| MRK | Marke | Brand | 0.53% |
| LDS | Landschaft | Landscape / Region | 0.37% |
| STR | StraBe | Street | 0.25% |
| AN | Anwalt | Lawyer | 0.21% |

## Architecture

### Batch Processing

The library implements dynamic batching to optimize inference throughput:

1. **BatchBuilder** collects input texts and groups them by similar length
2. Similar-length sequences are batched together to reduce padding
3. ONNX inference runs once per batch
4. Decoding runs in parallel using virtual threads

This approach significantly reduces padding overhead compared to fixed-size batching.

### Class Structure

- `GermanLerModel`: Singleton wrapper for ONNX model and tokenizer
- `GermanLerNer`: Main API for entity extraction
- `Batch`: Batched input container with padding
- `GermanLerModel.runInference()`: Single sequence inference
- `GermanLerModel.runBatchInference()`: Batched inference
- `GermanLerNer.extractEntitiesBatch()`: Full batch pipeline with dynamic batching

## Build

```bash
mvn clean compile
```

## Run Tests

```bash
mvn test
```

## Run Example

```bash
mvn exec:java -Dexec.mainClass="com.example.ner.Main"
```

## Maven Dependency

```xml
<dependency>
    <groupId>com.example</groupId>
    <artifactId>german-ler-onnx</artifactId>
    <version>1.0.0</version>
</dependency>
```

## License

This project uses the model from HuggingFace under its original license. See model card for details.