FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY server/ ./server/
COPY rl/ ./rl/

# V8: inference/training scripts available inside container for evaluators.
# environment.py replaces triage_environment.py (OpenEnv spec compliance).
# Note: rl/ training scripts require torch; install separately via requirements-rl.txt.
COPY inference.py policies.py rollout.py rewards.py client.py validate.py ./

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
