.PHONY: help venv install test proto run docker-build docker-up docker-down

help:
	@echo "OMR Python Service Makefile"
	@echo "  make install      安装依赖到 .venv"
	@echo "  make test         运行全部测试"
	@echo "  make proto        重新生成 protobuf 代码"
	@echo "  make run          启动服务"
	@echo "  make docker-build 构建 Docker 镜像"
	@echo "  make docker-up    启动 Docker Compose"
	@echo "  make docker-down  停止 Docker Compose"

venv:
	python -m venv .venv

install: venv
	.venv/Scripts/pip install -r requirements.txt

test:
	.venv/Scripts/python -m unittest discover -s omr_service/tests -p "test_*.py" -v

proto:
	bash omr_service/scripts/gen_proto.sh

run:
	.venv/Scripts/python -m omr_service.main

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
