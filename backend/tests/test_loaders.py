import subprocess
import sys
from pathlib import Path

from app.rag.loaders import LoaderOrchestrator


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
