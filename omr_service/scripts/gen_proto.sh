#!/usr/bin/env bash
# 重新生成 Python gRPC 代码
set -e

cd "$(dirname "$0")/../.."

python -m grpc_tools.protoc \
  -I./omr_service/rpc \
  --python_out=./omr_service/rpc \
  --grpc_python_out=./omr_service/rpc \
  ./omr_service/rpc/omr.proto

# 修复 Python 包内导入路径
sed -i 's/^import omr_pb2 as omr__pb2$/from omr_service.rpc import omr_pb2 as omr__pb2/' omr_service/rpc/omr_pb2_grpc.py

echo "protobuf 代码已生成并修复导入"
