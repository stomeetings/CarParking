# syntax=docker/dockerfile:1
# YOLO Smart Parking: Streamlit UI + detection (CPU). Build: docker compose build  (or docker build -t carparking .)
# One image, three roles (docker-compose.yml): the default command is the Streamlit page; `python scripts/worker.py` is
# the background monitoring worker; `uvicorn --factory api.main:create_app` the FastAPI backend (+ React app). Run
# alone (docker run), the page starts the worker itself (CARPARK_WORKER=embedded).
# Stages: web (React app: tsc, vitest, vite build) + base (dependencies + code) -> test (pytest + pyflakes) -> runtime;
# any failing test fails the build.
# Data (history DB, fixed slot maps, saved frames) is never in the image: mount a folder at /data.
# The camera URL (with its password) comes from the environment (CARPARK_RTSP_URL, e.g. docker compose's .env).

# React app (frontend/): type check, unit tests, production build; only dist/ goes into the runtime image
FROM node:24.18.0-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npx tsc --noEmit && npx vitest run && npx vite build


FROM python:3.10-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# CPU-only torch from the official CPU wheel index (the default PyPI wheel bundles ~2 GB of CUDA libraries). The +cpu
# pins make sure no CUDA build is picked; PyPI as the extra index serves torch's plain dependencies (the CPU index
# carries some of them only as source packages that fail to build there, e.g. typing-extensions)
RUN pip install torch==2.14.1+cpu torchvision==0.29.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# ultralytics / sahi pull in the GUI opencv-python; replace it with the headless build of the same version
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install --no-deps --force-reinstall "$(grep -i '^opencv-python-headless==' requirements.txt)" \
 && python -c "import cv2, torch, ultralytics, sahi, streamlit; print('cv2', cv2.__version__, 'torch', torch.__version__)"

COPY app.py yolov8s.pt ./
COPY models/ models/
COPY src/ src/
COPY scripts/ scripts/
COPY api/ api/


FROM base AS test
# throwaway paths: the tests never touch /data
ENV CARPARK_HISTORY_DB=/tmp/test/h.db \
    CARPARK_FIXED_SLOTS=/tmp/test/cfg/fixed_slots.json \
    CARPARK_DATASET_RAW=/tmp/test/raw
COPY tests/ tests/
RUN pip install pytest==8.4.2 pyflakes==3.4.0 httpx==0.28.1 \
 && python -m pytest -q -p no:cacheprovider tests \
 && python -m pyflakes src scripts tests api \
 && touch /tests-passed


FROM base AS runtime
# BuildKit builds only the stages the target needs: this copy makes the runtime image depend on the test stage
COPY --from=test /tests-passed /tests-passed
COPY --from=web /web/dist frontend/dist
ENV CARPARK_HISTORY_DB=/data/parking_history.db \
    CARPARK_FIXED_SLOTS=/data/config/fixed_slots.json \
    CARPARK_DATASET_RAW=/data/dataset_raw \
    CARPARK_TZ=Asia/Karachi \
    CARPARK_WORKER=embedded \
    HOME=/home/app
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data/config /data/dataset_raw \
 && chown -R app:app /data /app
USER app
VOLUME ["/data"]
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
