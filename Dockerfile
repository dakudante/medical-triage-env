FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY server/ ./server/

# V5: inference/training scripts copied for completeness.
# The container CMD only starts the server, but having these files in the
# image means an evaluator can run inference.py directly inside the container
# without needing a separate volume mount.
COPY inference.py policies.py rollout.py rewards.py client.py ./

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]