# NIS on-demand zoom viewer.
# Build context must be the package dir (nis_ondemand_viewer/), e.g.:
#   docker build -f deploy/Dockerfile -t nis-ondemand-viewer ..
# (docker-compose.yml in this dir already sets context: ..)

FROM python:3.11-slim

# libvips powers the preferred partial-TIFF backend (pyvips). The other
# backends (tifffile/zarr, pillow) are pure-python fallbacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libvips42 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# CPU-only torch keeps the image small; the service only torch.loads tensors.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Package lands at /app/nis_ondemand_viewer so `nis_ondemand_viewer.app:app` imports.
COPY . /app/nis_ondemand_viewer/

EXPOSE 8090
CMD ["uvicorn", "nis_ondemand_viewer.app:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "2"]
