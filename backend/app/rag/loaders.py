import json, os
from pathlib import Path  
from typing import List, Dict, Type  
from langchain_core.documents import Document  
from langchain_community.document_loaders import (  
    TextLoader,  
    CSVLoader,  
    UnstructuredMarkdownLoader,  
    UnstructuredPowerPointLoader,  
    UnstructuredExcelLoader,  
    Docx2txtLoader,  
)  
import pdfplumber  
import xlrd

  
  
class PDFPlumberLoader:  
    """Custom PDF loader using pdfplumber (mirrors services/file_parsers.py PDFParser)."""  
  
    def __init__(self, file_path: str):  
        self.file_path = file_path  
  
    def load(self) -> List[Document]:  
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

    def load(self) -> List[Document]:
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
        self.loaders: Dict[str, Type] = {  
            "pdf": PDFPlumberLoader,  
            "txt": TextLoader,  
            "md": UnstructuredMarkdownLoader,  
            "csv": CSVLoader,  
            "pptx": UnstructuredPowerPointLoader,  
            "xlsx": UnstructuredExcelLoader,  
            "xls": XLSLoader,
            "docx": Docx2txtLoader,  
        }  
  
    def register(self, extensions: List[str], loader_cls):  
        for ext in extensions:  
            self.loaders[ext.lower()] = loader_cls  
  
    def load_file(self, file_path: str) -> List[Document]:  
        ext = Path(file_path).suffix.lstrip(".").lower()  
        loader_cls = self.loaders.get(ext)  
        if not loader_cls:  
            raise ValueError(f"Unsupported file type: {ext}")  
        return loader_cls(file_path).load()  
  
    def load_directory(self, dir_path: str) -> List[Document]:  
        docs = []  
        for path in Path(dir_path).rglob("*"):  
            if path.suffix.lstrip(".").lower() in self.loaders:  
                docs.extend(self.load_file(str(path)))  
        return docs


if __name__ == "__main__":
    folder_path = r"C:\Users\kkrch\Local_project\ENTERPRISE-RAG-ASSISTANT\ENTERPRISE-RAG-ASSISTANT\backend\data\documents"  # Your input folder
    output_file = "loaded_documents.json"

    orchestrator = LoaderOrchestrator()

    try:
        documents = orchestrator.load_directory(folder_path)

        print(f"\nSuccessfully loaded {len(documents)} documents.\n")

        # Convert LangChain Documents into JSON-compatible dictionaries
        data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]

        # Save to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Successfully saved documents to: {output_file}")

        # Display preview
        for i, doc in enumerate(documents, start=1):
            print(f"\n{'=' * 60}")
            print(f"Document {i}")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Content length: {len(doc.page_content)} characters")
            print(f"\nContent Preview:\n")
            print(doc.page_content[:500])
            print(f"{'=' * 60}")

    except Exception as e:
        print(f"Error loading documents: {e}")