FROM docker.io/python:3.14 AS builder

WORKDIR /build

COPY pyproject.toml ./
COPY scnf/ scnf/

RUN pip install --no-cache-dir .

FROM docker.io/python:3.14

RUN pip install --no-cache-dir pytest

COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY tests/ tests/

ENV JAX_PLATFORMS=cpu

CMD ["python"]
