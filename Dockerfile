FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests

CMD ["python", "src/smoke_test.py", "--device", "cpu"]
