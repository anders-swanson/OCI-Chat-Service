package com.example.chat;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import com.oracle.bmc.auth.BasicAuthenticationDetailsProvider;
import com.oracle.bmc.generativeaiinference.GenerativeAiInference;
import com.oracle.bmc.generativeaiinference.GenerativeAiInferenceClient;
import com.oracle.bmc.generativeaiinference.model.BaseChatRequest;
import com.oracle.bmc.generativeaiinference.model.BaseChatResponse;
import com.oracle.bmc.generativeaiinference.model.ChatChoice;
import com.oracle.bmc.generativeaiinference.model.ChatContent;
import com.oracle.bmc.generativeaiinference.model.ChatDetails;
import com.oracle.bmc.generativeaiinference.model.CohereChatRequest;
import com.oracle.bmc.generativeaiinference.model.CohereChatResponse;
import com.oracle.bmc.generativeaiinference.model.CohereMessage;
import com.oracle.bmc.generativeaiinference.model.GenericChatRequest;
import com.oracle.bmc.generativeaiinference.model.GenericChatResponse;
import com.oracle.bmc.generativeaiinference.model.Message;
import com.oracle.bmc.generativeaiinference.model.ServingMode;
import com.oracle.bmc.generativeaiinference.model.TextContent;
import com.oracle.bmc.generativeaiinference.model.UserMessage;
import com.oracle.bmc.generativeaiinference.requests.ChatRequest;
import com.oracle.bmc.generativeaiinference.responses.ChatResponse;
import lombok.Builder;
import lombok.Getter;

/**
 * OCI GenAI Chat
 */
public class OCIChatService {
    private final GenerativeAiInference client;
    private final ServingMode servingMode;
    private final String compartment;
    private final String preambleOverride;
    private final Double temperature;
    private final Double frequencyPenalty;
    private final Integer maxTokens;
    private final Double presencePenalty;
    private final Double topP;
    private final Integer topK;
    private final InferenceRequestType inferenceRequestType;
    private List<CohereMessage> cohereChatMessages;
    private List<ChatChoice> genericChatMessages;

    @Builder
    public OCIChatService(BasicAuthenticationDetailsProvider authProvider,
                          ServingMode servingMode,
                          String compartment,
                          String preambleOverride,
                          String promptTemplate,
                          Double temperature,
                          Double frequencyPenalty,
                          Integer maxTokens,
                          Double presencePenalty,
                          Double topP,
                          Integer topK,
                          InferenceRequestType inferenceRequestType) {
        this.client = GenerativeAiInferenceClient.builder()
                .build(authProvider);
        this.servingMode = servingMode;
        this.compartment = compartment;
        this.preambleOverride = preambleOverride;

        this.temperature = Objects.requireNonNullElse(temperature, 1.0);
        this.frequencyPenalty = Objects.requireNonNullElse(frequencyPenalty, 0.0);
        this.maxTokens = Objects.requireNonNullElse(maxTokens, 600);
        this.presencePenalty = Objects.requireNonNullElse(presencePenalty, 0.0);
        this.topP = Objects.requireNonNullElse(topP, 0.75);
        this.inferenceRequestType = Objects.requireNonNullElse(inferenceRequestType, InferenceRequestType.COHERE);
        this.topK = Objects.requireNonNullElseGet(topK, () -> {
            if (this.inferenceRequestType == InferenceRequestType.COHERE) {
                return 0;
            }
            return -1;
        });
    }

    @Getter
    public enum InferenceRequestType {
        COHERE("COHERE"),
        LLAMA("LLAMA");

        private final String type;

        InferenceRequestType(String type) {
            this.type = type;
        }

    }

    /**
     * Chat using OCI GenAI.
     * @param prompt Prompt text sent to OCI GenAI chat model.
     * @return OCI GenAI ChatResponse
     */
    public String chat(String prompt) {
        ChatDetails chatDetails = ChatDetails.builder()
                .compartmentId(compartment)
                .servingMode(servingMode)
                .chatRequest(createChatRequest(prompt))
                .build();
        ChatRequest chatRequest = ChatRequest.builder()
                .body$(chatDetails)
                .build();
        ChatResponse response = client.chat(chatRequest);
        saveChatHistory(response);
        return extractText(response);
    }

    /**
     * Extract text from an OCI GenAI ChatResponse.
     * @param chatResponse The response to extract text from.
     * @return The chat response text.
     */
    private String extractText(ChatResponse chatResponse) {
        BaseChatResponse baseChatResponse = chatResponse.getChatResult().getChatResponse();
        if (baseChatResponse instanceof CohereChatResponse) {
            return ((CohereChatResponse) baseChatResponse).getText();
        } else if (baseChatResponse instanceof GenericChatResponse) {
            List<ChatChoice> choices =  ((GenericChatResponse) baseChatResponse).getChoices();
            List<ChatContent> contents = choices.get(choices.size() - 1).getMessage().getContent();
            ChatContent content = contents.get(contents.size() - 1);
            if (content instanceof TextContent) {
                return ((TextContent) content).getText();
            }
        }
        throw new IllegalStateException("Unexpected chat response type: " + baseChatResponse.getClass().getName());
    }

    /**
     * Create a ChatRequest from a text prompt. Supports COHERE or LLAMA inference types.
     * @param prompt To create a ChatRequest from.
     * @return A COHERE or LLAMA ChatRequest.
     */
    private BaseChatRequest createChatRequest(String prompt) {
        switch (inferenceRequestType) {
            case COHERE:
                return CohereChatRequest.builder()
                        .frequencyPenalty(frequencyPenalty)
                        .maxTokens(maxTokens)
                        .presencePenalty(presencePenalty)
                        .message(prompt)
                        .temperature(temperature)
                        .topP(topP)
                        .topK(topK)
                        .chatHistory(cohereChatMessages)
                        .preambleOverride(preambleOverride)
                        .build();
            case LLAMA:
                List<Message> messages = genericChatMessages == null ? new ArrayList<>() : genericChatMessages.stream()
                        .map(ChatChoice::getMessage)
                        .collect(Collectors.toList());
                ChatContent content = TextContent.builder()
                        .text(prompt)
                        .build();
                List<ChatContent> contents = new ArrayList<>();
                contents.add(content);
                UserMessage message = UserMessage.builder()
                        .name("USER")
                        .content(contents)
                        .build();
                messages.add(message);
                return GenericChatRequest.builder()
                        .messages(messages)
                        .frequencyPenalty(frequencyPenalty)
                        .temperature(temperature)
                        .maxTokens(maxTokens)
                        .presencePenalty(presencePenalty)
                        .topP(topP)
                        .topK(topK)
                        .build();
        }
        throw new IllegalArgumentException("Unsupported inference request type: " + inferenceRequestType);
    }


    /**
     * Save the current chat history to memory.
     * @param chatResponse The latest chat response.
     */
    private void saveChatHistory(ChatResponse chatResponse) {
        BaseChatResponse baseChatResponse = chatResponse.getChatResult().getChatResponse();
        if (baseChatResponse instanceof CohereChatResponse) {
            cohereChatMessages = ((CohereChatResponse) baseChatResponse).getChatHistory();
        } else if (baseChatResponse instanceof GenericChatResponse) {
            genericChatMessages = ((GenericChatResponse) baseChatResponse).getChoices();
        } else {
            throw new IllegalStateException("Unexpected chat response type: " + baseChatResponse.getClass().getName());
        }
    }
}