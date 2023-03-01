FROM python:3.11-slim-buster

EXPOSE 8080
RUN pip install langchain faiss-cpu openai fastapi "uvicorn[standard]"
WORKDIR /app
COPY ./ /app
# uncomment these lines to generate your index at container build time
# ENV OPENAI_API_KEY=sk-etc
# RUN python -c "import microllama; microllama.get_index()"
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]