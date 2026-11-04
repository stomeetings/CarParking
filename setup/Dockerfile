# The setup service: SAM 2.1 + GeoCalib + Depth Anything V2 behind a small HTTP API (setup/service.py), run on demand
# (docker compose --profile setup up -d setup). A separate image on purpose: none of this (torch models, transformers,
# the weights, ~1 GB) is in the worker / api / ui image, and none of it is loaded while the live system runs.
#   docker compose --profile setup build setup
# Python 3.12 (the main image is 3.10): current kornia / matplotlib releases need >= 3.11; this image shares no environment with it
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/torch \
    SAM2_BUILD_CUDA=0 \
    SAM2_BUILD_ALLOW_ERRORS=1
WORKDIR /app

RUN pip install torch==2.14.1+cpu torchvision==0.29.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

COPY setup/requirements.txt setup/manifest.json setup/fetch_weights.py ./setup/
# SAM 2 and GeoCalib from their pinned commits (a commit is content-addressed); --no-deps: GeoCalib's own requirements
# name the GUI opencv-python, and everything they need is in setup/requirements.txt
RUN pip install -r setup/requirements.txt \
 && pip install --no-deps \
        https://github.com/facebookresearch/sam2/archive/2b90b9f5ceec907a1c18123530e92e794ad901a4.tar.gz \
        https://github.com/cvg/GeoCalib/archive/97b8968e7798a66bf04fcf791fb535624241bda7.tar.gz \
 && python -c "import cv2, torch, transformers, kornia, sam2, geocalib; print('cv2', cv2.__version__, 'torch', torch.__version__)"

# the weights, verified against sha256 (the build fails on a mismatch); GeoCalib looks for its file in torch's hub cache
RUN python setup/fetch_weights.py setup/manifest.json /opt/weights \
 && mkdir -p /opt/torch/hub/geocalib \
 && cp /opt/weights/geocalib/geocalib-pinhole.tar /opt/torch/hub/geocalib/pinhole.tar

# the numpy-only geometry code shared with the api (surface / calibration helpers)
COPY src/carparking/__init__.py src/carparking/__init__.py
COPY src/carparking/geometry/ src/carparking/geometry/
COPY setup/models.py setup/service.py ./setup/
COPY setup/test_service.py ./setup/
ENV PYTHONPATH=/app/src:/app \
    SETUP_WEIGHTS=/opt/weights
# the service's HTTP contract with fake models (fails the build if it breaks); pytest is not kept
RUN pip install pytest==8.4.2 httpx==0.28.1 && python -m pytest -q -p no:cacheprovider setup/test_service.py \
 && pip uninstall -y pytest
RUN useradd --create-home --uid 1000 app && chown -R app:app /app /opt/torch
USER app
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=4)"
CMD ["uvicorn", "setup.service:app", "--host", "0.0.0.0", "--port", "8100"]
