import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.document_loaders import BaseLoader

from app.rag.loaders import LoaderOrchestrator, RecursiveDirectoryLoader


def test_loader_import_does_not_use_sunset_package():
    backend = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable, "-W", "default", "-c",
            "import sys; import app.rag.loaders; "
            "assert not any(name == 'langchain_community' or name.startswith('langchain_community.') "
            "for name in sys.modules)",
        ],
        cwd=backend, capture_output=True, text=True, check=True,
    )
    assert "langchain-community is being sunset" not in result.stderr


def test_text_and_csv_loader_content_and_row_metadata(tmp_path):
    loader = LoaderOrchestrator()
    text = tmp_path / "notes.txt"
    text.write_text("first line\nsecond line")
    docs = loader.load_file(str(text))
    assert [(doc.page_content, doc.metadata) for doc in docs] == [
        ("first line\nsecond line", {"source": str(text)})
    ]

    csv_file = tmp_path / "rows.csv"
    csv_file.write_text('name,description\n Alice ," hello, world "\nBob,second\n')
    docs = loader.load_file(str(csv_file))
    assert [doc.page_content for doc in docs] == [
        "name: Alice\ndescription: hello, world",
        "name: Bob\ndescription: second",
    ]
    assert [doc.metadata for doc in docs] == [
        {"source": str(csv_file), "row": 0}, {"source": str(csv_file), "row": 1}
    ]


def test_recursive_langchain_loader_handles_nested_empty_and_missing(tmp_path):
    loader = RecursiveDirectoryLoader(tmp_path)
    assert isinstance(loader, BaseLoader)
    assert loader.load() == []
    with pytest.raises(FileNotFoundError, match="Document directory does not exist"):
        RecursiveDirectoryLoader(tmp_path / "missing").load()
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "root.txt").write_text("root content")
    (nested / "child.txt").write_text("nested content")
    (nested / "ignore.xyz").write_text("unsupported")
    (nested / "directory.txt").mkdir()
    documents = asyncio.run(loader.aload())
    assert {doc.page_content for doc in documents} == {"root content", "nested content"}
    assert {doc.metadata["source"] for doc in documents} == {
        str(tmp_path / "root.txt"), str(nested / "child.txt"),
    }


def test_standalone_loader_finds_backend_documents_without_overwriting(tmp_path):
    source_backend = Path(__file__).resolve().parents[1]
    backend = tmp_path / "backend"
    rag = backend / "app" / "rag"
    core = backend / "app" / "core"
    rag.mkdir(parents=True)
    core.mkdir()
    (backend / "app" / "__init__.py").touch()
    (core / "__init__.py").touch()
    shutil.copyfile(source_backend / "app" / "rag" / "loaders.py", rag / "loaders.py")
    shutil.copyfile(source_backend / "app" / "core" / "config.py", core / "config.py")
    document_dir = backend / "data" / "documents"
    document_dir.mkdir(parents=True)
    (document_dir / "notes.txt").write_text("example content")
    existing = rag / "loaded_documents.json"
    existing.write_text("existing output")
    env = os.environ.copy()
    env.pop("DOCUMENTS_DIR", None)
    result = subprocess.run(
        [sys.executable, "loaders.py"], cwd=rag, env=env,
        capture_output=True, text=True, check=True,
    )
    assert str(document_dir) in result.stdout
    assert "Successfully loaded 1 documents" in result.stdout
    assert existing.read_text() == "existing output"
    generated = list(rag.glob("loaded_documents-*.json"))
    assert len(generated) == 1
    assert json.loads(generated[0].read_text(encoding="utf-8"))[0]["page_content"] == "example content"


def test_standalone_loader_reports_empty_and_missing_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "app" / "rag" / "loaders.py"
    env = os.environ.copy()
    directory = tmp_path / "documents"
    env["DOCUMENTS_DIR"] = str(directory)
    missing = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert missing.returncode != 0
    assert str(directory) in missing.stderr
    assert not list(tmp_path.glob("loaded_documents*.json"))
    directory.mkdir()
    empty = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert empty.returncode == 0
    assert str(directory) in empty.stdout
    assert "No extractable documents found" in empty.stdout
    assert not list(tmp_path.glob("loaded_documents*.json"))
