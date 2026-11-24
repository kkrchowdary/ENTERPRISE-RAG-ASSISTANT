from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


def test_health(monkeypatch):
    async def no_database():
        pass

    monkeypatch.setattr(main_module, "initialize_database", no_database)
    monkeypatch.setattr(main_module, "dispose_database", no_database)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}