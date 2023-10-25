FROM python:3.11-slim-buster

EXPOSE 8080
RUN pip install microllama
WORKDIR /app
COPY ./ /app
# Remove these two lines if you don't want to
# generate your index at container build time
ENV OPENAI_API_KEY={openai_api_key}
# Uncomment this line to enable debug mode
# ENV DEBUG=True
# Uncomment this line to use a different model from gpt-3.5-turbo
# ENV MODEL=gpt-4
RUN microllama index
CMD ["microllama"]