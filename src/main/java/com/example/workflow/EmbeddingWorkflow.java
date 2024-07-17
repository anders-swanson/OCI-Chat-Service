package com.example.workflow;

import com.example.documentloader.OCIDocumentLoader;
import com.example.embeddingmodel.OCIEmbeddingModel;
import com.example.splitter.Splitter;
import com.example.vectorstore.OracleVectorStore;
import lombok.Builder;

@Builder
public class EmbeddingWorkflow implements Runnable {
    private final OracleVectorStore vectorStore;
    private final OCIEmbeddingModel embeddingModel;
    private final OCIDocumentLoader documentLoader;
    private final String namespace;
    private final String bucketName;
    private final String objectPrefix;
    private final Splitter<String> splitter;

    @Override
    public void run() {
        // Stream documents from OCI object storage.
        documentLoader.streamDocuments(bucketName, objectPrefix)
                // Split each object storage document into chunks.
                .map(splitter::split)
                // Embed each chunk list using OCI GenAI service.
                .map(embeddingModel::embedAll)
                // Store embeddings in Oracle Database 23ai.
                .forEach(vectorStore::addAll);
    }
}
