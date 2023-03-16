FROM python:3.11-slim-buster

EXPOSE 8080
RUN pip install microllama
WORKDIR /app
COPY ./ /app
# Remove these two lines if you don't want to
# generate your index at container build time
ENV OPENAI_API_KEY={openai_api_key}
RUN microllama index
CMD ["microllama"]