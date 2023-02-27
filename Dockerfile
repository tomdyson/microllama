FROM python:3.11-slim-buster

EXPOSE 8080
RUN pip install langchain faiss-cpu openai fastapi "uvicorn[standard]"
WORKDIR /app
COPY ./ /app
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]