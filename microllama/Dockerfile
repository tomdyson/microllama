FROM python:3.11-slim-buster

EXPOSE 8080
RUN pip install microllama
WORKDIR /app
COPY ./ /app

# --- Environment Variables --- 
# Set API keys and configuration via environment variables.
# OPENAI_API_KEY is *always* required for embeddings.
# Set other keys (e.g., GEMINI_API_KEY) if using a different chat model.
# Replace {openai_api_key} below or set the variable at runtime
ENV OPENAI_API_KEY={openai_api_key}
# Example for Gemini:
# ENV GEMINI_API_KEY=your_gemini_key

# Uncomment this line to enable debug mode
# ENV DEBUG=True

# Set the chat model ID used by llm. 
# Ensure the corresponding llm plugin is installed (e.g., llm-openai is included by default).
# ENV MODEL=gpt-4 
# ENV MODEL=claude-3-opus # Requires pip install llm-anthropic
# ENV MODEL=gemini-pro   # Requires pip install llm-gemini

# Generate FAISS index at build time (requires OPENAI_API_KEY to be set above)
# Remove this line if you prefer to index on first run or manually.
RUN microllama index

CMD ["microllama"]