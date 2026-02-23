import os
import shutil
import pytest
from httpx import AsyncClient, ASGITransport
from server import app

TEST_DATA_DIR = "/tmp/mem0_test_data"
os.environ["MEM0_DATA_DIR"] = TEST_DATA_DIR


def pytest_configure(config):
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)


def pytest_unconfigure(config):
    import server as server_mod
    server_mod.memory = None
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")
