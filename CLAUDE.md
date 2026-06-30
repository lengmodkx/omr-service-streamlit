# CLAUDE.md

## Overview

This repository is a Python-based OMR (Optical Mark Recognition) microservice.
It replaces the previous Streamlit demo and the abandoned Go service direction.

The service exposes:
- **Dubbo Triple / gRPC** endpoints for template parsing, recognition, verification, and re-verification.
- **Redis Stream** consumer/producer for batch image recognition jobs.
- **Nacos** for both service registration/discovery and configuration management.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# Run service
python -m omr_service.main

# Regenerate protobuf (after editing omr_service/rpc/omr.proto)
python -m grpc_tools.protoc -I./omr_service/rpc --python_out=./omr_service/rpc --grpc_python_out=./omr_service/rpc ./omr_service/rpc/omr.proto
# Then fix the import in omr_pb2_grpc.py:
#   import omr_pb2 as omr__pb2
# -> from omr_service.rpc import omr_pb2 as omr__pb2

# Run tests
python -m unittest discover -s omr_service/tests -p "test_*.py" -v

# Docker
docker compose build
docker compose up -d
```

## Configuration

Configuration priority: **Nacos > environment variables > defaults**.

Create a Nacos config:
- dataId: `omr-service.yaml`
- group: `DEFAULT_GROUP`

Example:

```yaml
nacos_server: 127.0.0.1:8848
nacos_namespace: public
redis:
  host: 47.99.83.217
  port: 6379
  password: your_password
  db: 1
omr_worker_count: 4
```

The loader in `config.py` flattens nested YAML keys (e.g. `redis.host`).

## Architecture

### gRPC Service (`omr_service/rpc/omr_service.py`)

The four RPC methods wrap the engine.

### Redis Batch Flow (`omr_service/mq/`)

- `consumer.py` reads from Redis Stream `omr:batch:job` using a consumer group.
- `job_handler.py` processes jobs concurrently and writes results to `omr:batch:result`.

### Nacos (`omr_service/nacos_config.py`, `omr_service/nacos_reg.py`)

- Config client pulls config at startup and optionally listens for changes.
- Registrator registers the instance with Dubbo Triple metadata under both app and interface names.

## Key Implementation Details

### Coordinate Scaling

Template coordinates are based on reference image dimensions and scaled at runtime.

### Golden Template Grid Generation

`StandardTemplate._generate_grid()` supports `option_axis` and `reverse_q`.

### Windows Chinese Path Handling

`cv2.imwrite()` has UTF-8 issues on Windows. The engine uses `cv2.imencode + open(filepath, 'wb')`.

## Adding New RPC Methods

1. Update `omr_service/rpc/omr.proto`.
2. Regenerate Python gRPC code.
3. Implement the method in `omr_service/rpc/omr_service.py`.
4. Add a test in `omr_service/tests/test_rpc.py`.
