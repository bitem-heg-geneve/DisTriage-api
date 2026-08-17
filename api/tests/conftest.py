import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope='module')
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
