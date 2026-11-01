FROM python:3.12-slim

WORKDIR /app

COPY docker-requirements.txt .

RUN pip install \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --no-cache-dir \
    --default-timeout=300 \
    --retries 10 \
    -r docker-requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
