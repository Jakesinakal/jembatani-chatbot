import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


def setup_rag() -> Chroma:
    """Inisialisasi RAG: load dokumen panduan, embed, simpan ke ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    persist_dir = "./chroma_db"

    # Jika sudah ada vector store, load langsung
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        return vectorstore

    # Buat vector store baru dari dokumen panduan
    doc_path = os.path.join(os.path.dirname(__file__), "docs", "panduan_jembatantani.txt")
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def query_rag(vectorstore: Chroma, query: str, k: int = 3) -> str:
    """Ambil konteks relevan dari vector store berdasarkan query."""
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)
