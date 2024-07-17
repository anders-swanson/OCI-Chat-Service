package com.example.workflow;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import com.example.chat.OCIChatService;
import com.example.embeddingmodel.OCIEmbeddingModel;
import com.example.model.Embedding;
import com.example.vectorstore.OracleVectorStore;
import com.example.vectorstore.SearchRequest;
import lombok.Builder;

@Builder
public class ChatWorkflow implements Callable<String> {
    private final OracleVectorStore vectorStore;
    private final OCIChatService chatService;
    private final OCIEmbeddingModel embeddingModel;

    private final String userQuestion;
    private final String promptTemplate;
    private final double minScore;

    @Override
    public String call() throws Exception {
        // Embed the user question.
        Embedding embedding = embeddingModel.embed(userQuestion);
        // Create a search request using the user question's embedding.
        SearchRequest searchRequest = SearchRequest.builder()
                .text(embedding.content())
                .vector(embedding.vector())
                .maxResults(5)
                .minScore(minScore)
                .build();
        // Search the vector store for any content related to the user question.
        List<Embedding> results = vectorStore.search(searchRequest);
        // If related content was found, add it to the chat service prompt.
        String context = results.isEmpty() ?
                "No additional context was provided." :
                results.stream()
                        .map(Embedding::content)
                        .collect(Collectors.joining(", "));;
        String prompt = String.format(promptTemplate, userQuestion, context);
        // Call the chat service using the content-enriched prompt.
        return chatService.chat(prompt);
    }
}
