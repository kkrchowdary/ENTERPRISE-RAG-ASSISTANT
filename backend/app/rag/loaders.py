import csv
import json
from collections.abc import Iterator
from pathlib import Path

import docx2txt
import pdfplumber
import xlrd
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class TextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        return [Document(page_content=Path(self.file_path).read_text(),
                        metadata={"source": self.file_path})]


class CSVLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        with open(self.file_path, newline="") as file:
            documents = []
            for index, row in enumerate(csv.DictReader(file)):
                fields = []
                for key, value in row.items():
                    if isinstance(value, str):
                        value = value.strip()
                    elif isinstance(value, list):
                        value = ",".join(item.strip() for item in value)
                    fields.append(f"{key.strip() if key is not None else key}: {value}")
                documents.append(Document(
                    page_content="\n".join(fields),
                    metadata={"source": self.file_path, "row": index},
                ))
            return documents


class DocxLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        return [Document(page_content=docx2txt.process(self.file_path),
                            metadata={"source": self.file_path})]


class UnstructuredLocalLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        suffix = Path(self.file_path).suffix.lower()
        if suffix == ".md":
            from unstructured.partition.md import partition_md
            elements = partition_md(filename=self.file_path)
        elif suffix == ".pptx":
            from unstructured.partition.pptx import partition_pptx
            elements = partition_pptx(filename=self.file_path)
        else:
            from unstructured.partition.xlsx import partition_xlsx
            elements = partition_xlsx(filename=self.file_path)
        return [Document(page_content="\n\n".join(str(element) for element in elements),
                            metadata={"source": self.file_path})]


class PDFPlumberLoader:  
    """Custom PDF loader using pdfplumber (mirrors services/file_parsers.py PDFParser)."""  
    
    def __init__(self, file_path: str):  
        self.file_path = file_path  
    
    def load(self) -> list[Document]:
        docs = []  
        with pdfplumber.open(self.file_path) as pdf:  
            for i, page in enumerate(pdf.pages):  
                text = page.extract_text() or ""  
                docs.append(Document(  
                    page_content=text,  
                    metadata={"source": self.file_path, "page": i},  
                ))  
        return docs  
  
  
class XLSLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        workbook = xlrd.open_workbook(self.file_path)
        return [
            Document(
                page_content="\n".join(
                    "\t".join(str(value) for value in sheet.row_values(row))
                    for row in range(sheet.nrows)
                ),
                metadata={"source": self.file_path},
            )
            for sheet in workbook.sheets()
        ]


class LoaderOrchestrator:  
    """Registry that maps file extensions to LangChain loaders."""  
  
    def __init__(self):  
        self.loaders: dict[str, type] = {
            "pdf": PDFPlumberLoader,  
            "txt": TextLoader,  
            "md": UnstructuredLocalLoader,
            "csv": CSVLoader,  
            "pptx": UnstructuredLocalLoader,
            "xlsx": UnstructuredLocalLoader,
            "xls": XLSLoader,
            "docx": DocxLoader,
        }  
  
    def register(self, extensions: list[str], loader_cls: type) -> None:
        for ext in extensions:  
            self.loaders[ext.lower()] = loader_cls  
  
    def load_file(self, file_path: str) -> list[Document]:
        ext = Path(file_path).suffix.lstrip(".").lower()  
        loader_cls = self.loaders.get(ext)  
        if not loader_cls:  
            raise ValueError(f"Unsupported file type: {ext}")  
        return loader_cls(file_path).load()  
  
    def load_directory(self, dir_path: str) -> list[Document]:
        return RecursiveDirectoryLoader(dir_path, self).load()


class RecursiveDirectoryLoader(BaseLoader):
    def __init__(self, directory: str | Path, orchestrator: LoaderOrchestrator | None = None):
        self.directory = Path(directory)
        self.orchestrator = orchestrator or LoaderOrchestrator()

    def lazy_load(self) -> Iterator[Document]:
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Document directory does not exist: {self.directory}")
        for path in self.directory.rglob("*"):
            if path.is_file() and path.suffix.lstrip(".").lower() in self.orchestrator.loaders:
                yield from self.orchestrator.load_file(str(path))


if __name__ == "__main__":
    import sys
    from uuid import uuid4

    backend_dir = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(backend_dir))
    from app.core.config import Settings

    folder_path = Settings().documents_dir
    if not folder_path.is_absolute():
        folder_path = backend_dir / folder_path
    folder_path = folder_path.resolve()
    print(f"Reading documents from: {folder_path}")
    documents = LoaderOrchestrator().load_directory(str(folder_path))
    print(f"\nSuccessfully loaded {len(documents)} documents.\n")

    if not documents:
        print("No extractable documents found; no output file was created.")
        sys.exit(0)

    output_file = Path("loaded_documents.json")
    if output_file.exists():
        output_file = Path(f"loaded_documents-{uuid4().hex}.json")

    # Convert LangChain Documents into JSON-compatible dictionaries
    data = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]

    # Save to JSON file
    with output_file.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Successfully saved documents to: {output_file}")

    # Display preview
    for i, doc in enumerate(documents, start=1):
        print(f"\n{'=' * 60}")
        print(f"Document {i}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content length: {len(doc.page_content)} characters")
        print("\nContent Preview:\n")
        print(doc.page_content[:500])
        print(f"{'=' * 60}")