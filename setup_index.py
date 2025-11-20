import os
from dotenv import load_dotenv

from typing import List, Iterable

from pypdf import PdfReader

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ use community imports for v0.2+ LangChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    
# ---------- Custom loader that never touches page_labels ----------


class SafePyPDFLoader(PyPDFLoader):

    def lazy_load(self) -> Iterable[Document]:
        reader = PdfReader(self.file_path)
        total_pages = len(reader.pages)

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            metadata = {
                "source": self.file_path,
                "page": i,              # numeric index
                "total_pages": total_pages,
            }
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        return list(self.lazy_load())


def load_pdf(path: str) -> List[Document]:
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=SafePyPDFLoader,   
    )
    documents = loader.load()
    return documents


def filter_docs(docs: List[Document]) -> List[Document]:
    min_doc = []

    for doc in docs:
        page_content = doc.page_content
        total_pages = doc.metadata.get("total_pages")
        page = doc.metadata.get("page")
        source = doc.metadata.get("source")

        min_doc.append(
            Document(
                metadata={
                    "total_pages": total_pages,
                    "page": page,
                    "source": source,
                },
                page_content=page_content,
            )
        )

    return min_doc


def text_chunk(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(docs)


def load_embedding():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

pc = Pinecone(api_key = PINECONE_API_KEY)

def create_index(index_name):
    
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type='dense',
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    


if __name__ == "__main__":
    documents = load_pdf("data")
    filtered_docs = filter_docs(documents)
    text_chunks = text_chunk(filtered_docs)
    embedding = load_embedding()
    index_name = 'legal'

    if pc.has_index(index_name):
        pc.delete_index(index_name)

    create_index(index_name)

    index = pc.Index(index_name)

    vectorstore = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embedding,
    )


