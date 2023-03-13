FROM python:3.11-slim-buster

EXPOSE 8080
RUN pip install microllama
WORKDIR /app
COPY ./ /app
# uncomment these lines to generate your index at container build time
# ENV OPENAI_API_KEY=sk-etc
# RUN ml_index
CMD ["ml_serve"]