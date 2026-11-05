PYTHON ?= .venv/bin/python
PY := $(abspath $(PYTHON))
GEN_DIR := worker/leviathan_worker/_gen

.PHONY: check test-rust test-py test-engine test-rl proto-py fmt lint venv

check: test-rust test-py test-engine test-rl

test-rust:
	cargo test --workspace

test-py: proto-py
	cd worker && $(PY) -m pytest -q

# The engine's tests import the worker's hashing (convention check), hence PYTHONPATH.
test-engine:
	cd engine && PYTHONPATH=../worker $(PY) -m pytest -q

test-rl:
	cd rl && PYTHONPATH=.:../worker $(PY) -m pytest -q

proto-py:
	mkdir -p $(GEN_DIR)
	$(PY) -m grpc_tools.protoc -I proto \
		--python_out=$(GEN_DIR) --grpc_python_out=$(GEN_DIR) --pyi_out=$(GEN_DIR) \
		proto/leviathan/worker/v1/worker.proto

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets -- -D warnings

venv:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e 'worker[dev]'
