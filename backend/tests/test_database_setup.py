import asyncio

import pytest
from sqlalchemy.exc import SQLAlchemyError

import app.database.connection as database


class FakeConnection:
    def __init__(self, *, installed, available=True, deny=False, cascade="a"):
        self.installed = installed
        self.available = available
        self.deny = deny
        self.cascade = cascade
        self.statements = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def scalar(self, statement):
        sql = str(statement)
        if "pg_extension" in sql:
            return self.installed
        if "pg_available_extensions" in sql:
            return self.available
        if "pg_constraint" in sql:
            return self.cascade
        raise AssertionError(sql)

    async def execute(self, statement):
        sql = str(statement)
        self.statements.append(sql)
        if self.deny and "CREATE EXTENSION" in sql:
            raise SQLAlchemyError("permission denied")

    async def run_sync(self, fn):
        pass

    def begin(self):
        return self


def test_setup_enables_pgvector_and_upgrades_foreign_keys(monkeypatch):
    connection = FakeConnection(installed=False)
    monkeypatch.setattr(database, "get_engine", lambda: connection)
    asyncio.run(database.initialize_database())
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in sql for sql in connection.statements)
    assert any("ADD COLUMN IF NOT EXISTS chunk_index" in sql for sql in connection.statements)
    assert sum("ON DELETE CASCADE" in sql for sql in connection.statements) == 2
    connection = FakeConnection(installed=True, cascade="c")
    monkeypatch.setattr(database, "get_engine", lambda: connection)
    asyncio.run(database.initialize_database())
    assert not any("CREATE EXTENSION" in sql or "DROP CONSTRAINT" in sql
                   for sql in connection.statements)


@pytest.mark.parametrize(
    ("connection", "message"),
    [
        (FakeConnection(installed=False, available=False), "install pgvector"),
        (FakeConnection(installed=False, deny=True), "grant CREATE privileges"),
        (FakeConnection(installed=True, cascade=None), "Expected foreign key"),
    ],
)
def test_setup_fails_explicitly(monkeypatch, caplog, connection, message):
    monkeypatch.setattr(database, "get_engine", lambda: connection)
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(database.initialize_database())
    if "foreign key" not in message:
        assert message in caplog.text
