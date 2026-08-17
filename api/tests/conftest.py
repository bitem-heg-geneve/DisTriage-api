import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with patch("app.main.init_db", new_callable=AsyncMock), \
         patch("app.main.close_db", new_callable=AsyncMock):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

