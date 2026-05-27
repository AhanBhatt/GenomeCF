FROM python:3.13-slim

WORKDIR /workspace/GenomeCF
COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[benchmark,dev]

CMD ["python", "-m", "genomecf.cli", "reproduce-quickstart"]
