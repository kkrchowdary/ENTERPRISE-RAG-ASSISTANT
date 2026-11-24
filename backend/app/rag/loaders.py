import json
from pathlib import Path
from typing import List, Dict, Type, Any
import docx2txt
import pdfplumber
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFPlumberLoader:
    """Custom PDF loader using pdfplumber."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        docs = []
        with pdfplumber.open(self.file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                docs.append(Document(
                    page_content=text,
                    metadata={"source": self.file_path, "page": i + 1},
                ))
        return docs


class DocxLoader:
    """Custom DOCX loader using docx2txt."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        text = docx2txt.process(self.file_path) or ""
        return [Document(
            page_content=text,
            metadata={"source": self.file_path},
        )]


class UnstructuredLocalLoader:
    """Local partition loader for MD, PPTX, and XLSX."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
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

        return [Document(
            page_content="\n\n".join(str(el) for el in elements if str(el).strip()),
            metadata={"source": self.file_path},
        )]


class LoaderOrchestrator:
    """Registry that maps file extensions to loaders and chunks documents."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.loaders: Dict[str, Type] = {
            "pdf": PDFPlumberLoader,
            "txt": TextLoader,
            "md": UnstructuredLocalLoader,
            "csv": CSVLoader,
            "pptx": UnstructuredLocalLoader,
            "xlsx": UnstructuredLocalLoader,
            "xls": UnstructuredExcelLoader,
            "docx": DocxLoader,
        }
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def register(self, extensions: List[str], loader_cls):
        for ext in extensions:
            self.loaders[ext.lower()] = loader_cls

    def load_file(self, file_path: str) -> List[Document]:
        ext = Path(file_path).suffix.lstrip(".").lower()
        loader_cls = self.loaders.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader_cls(file_path).load()

    def load_directory_structured(self, dir_path: str) -> List[Dict[str, Any]]:
        structured_docs = []

        for path in Path(dir_path).rglob("*"):
            if not path.is_file():
                continue
            ext = path.suffix.lstrip(".").lower()
            if ext not in self.loaders:
                print(f"Skipping unsupported file: {path.name}")
                continue

            raw_docs = self.load_file(str(path))
            chunks = self.splitter.split_documents(raw_docs)

            structured_docs.append({
                "document_name": path.name,
                "file_type": ext,
                "source_path": str(path),
                "total_chunks": len(chunks),
                "chunks": [
                    {
                        "chunk_id": f"{path.stem}_chunk_{i}",
                        "chunk_index": i,
                        "page": chunk.metadata.get("page"),
                        "char_count": len(chunk.page_content),
                        "content": chunk.page_content,
                    }
                    for i, chunk in enumerate(chunks)
                ],
            })

        return structured_docs


if __name__ == "__main__":
    folder_path = r"C:\Users\kkrch\Local_project\ENTERPRISE-RAG-ASSISTANT\ENTERPRISE-RAG-ASSISTANT\backend\data\documents"
    output_file = r"C:\Users\kkrch\Local_project\ENTERPRISE-RAG-ASSISTANT\ENTERPRISE-RAG-ASSISTANT\backend\data\loaded_documents.json"

    orchestrator = LoaderOrchestrator(chunk_size=800, chunk_overlap=100)
    results = orchestrator.load_directory_structured(folder_path)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nSaved {len(results)} structured documents to {output_file}")
    for doc in results:
        print(f"  - {doc['document_name']} ({doc['file_type']}): {doc['total_chunks']} chunks")