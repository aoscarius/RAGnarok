import os
import sys
import time
from tqdm import tqdm

from llama_cpp import Llama
from langchain_chroma import Chroma
from langchain_text_splitters.markdown import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredFileLoader
)

class LlamaCppEmbedder:
    """ Simple llama-cpp embedded class implementation """
    def __init__(self, model_path, **kwargs):
        self.embed = Llama(
            model_path=model_path,
            embedding=True,
            **kwargs
        )

    def embed_documents(self, texts):
        embeddings = []

        with open(os.devnull, 'w') as ferr:
            _stderr = sys.stderr
            sys.stderr = ferr

            for t in tqdm(texts, desc="Embedding documents", unit="chunk", file=sys.stdout):
                emb = self.embed.create_embedding(t)["data"][0]["embedding"]
                embeddings.append(emb)

            sys.stderr = _stderr

        return embeddings

    def embed_query(self, text):
        return self.embed.create_embedding(text)["data"][0]["embedding"]

def loadFiles(filepath):
    """ Return a list of loaded document files """
    ext = filepath.lower()

    try:
        if ext.endswith(".md"):
            return UnstructuredMarkdownLoader(filepath).load()
        elif ext.endswith(".html") or ext.endswith(".htm"):
            return UnstructuredHTMLLoader(filepath).load()
        elif ext.endswith(".txt"):
            return UnstructuredFileLoader(filepath, mode="elements").load()
        else:
            return []

    except Exception as e:
        print(f"[ERROR] Cannot load {filepath}: {e}")
        return []

def documentRag(
    source_directory: str,
    db_path: str,
    model_path: str,
    collection_name: str = "documents",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
  ):
    """
    Creates or updates a vector database from accepted files in a specified folder.
    Only adds documents that are not already present in the database.

    Args:
        source_directory (str): The path to the folder containing accepted files.
        db_path (str): The directory to store the Chroma vector database.
        collection_name (str): The name to assign to knowledge collection.
        model_path (str): The path where GGUF models are stored.
        chunk_size (int): The size of each text chunk when splitting documents.
        chunk_overlap (int): The number of overlapping characters between chunks.
    """

    start_time = time.time()

    # Validate folders and path
    if not os.path.isdir(source_directory):
        raise NotADirectoryError(f"Source directory not found: {source_directory}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Embedding model not found at: {model_path}")

    print(f"[INFO] Loading embeddings model from: {model_path}")
    embeddings = LlamaCppEmbedder(
            model_path=model_path,
            n_threads=int(os.cpu_count() / 2),
            n_gpu_layers=0,
            temperature=0.2,
            top_p=0.9,
            max_tokens=32,
            n_ctx=512,
            verbose=False,
    )

    print(f"[INFO] Initialize recursive text splitter")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Create DB if missing
    if not os.path.exists(db_path):
        print(f"[INFO] Creating new Chroma DB at: {db_path}")

        all_docs = []
        supported_ext = (".md", ".txt", ".html", ".htm")
        print(f"[INFO] Supported scanning files: {supported_ext}")

        # Recursively scan docs directory
        for root, dirs, files in os.walk(source_directory):
            for filename in files:
                filepath = os.path.join(root, filename).replace("\\", "/")
                if filename.lower().endswith(supported_ext):
                    docs = loadFiles(filepath)

                    if docs:
                        print(f"[OK] Loaded file: {filename} → {len(docs)} docs")
                        all_docs.extend(docs)
                    else:
                        print(f"[WARN] Skipped file (no docs): {filename}")

        if not all_docs:
            raise ValueError(f"No {supported_ext} files could be loaded.")

        # Generate chunks using text_splitters
        chunks = text_splitter.split_documents(all_docs)
        # Clean empty chunks
        chunks = [c for c in chunks if c.page_content.strip()]
        # Remove metadata that Chroma does not support
        chunks = filter_complex_metadata(chunks)

        if not chunks:
            raise ValueError("Document chunks are empty after splitting!")

        print(f"[INFO] Adding {len(chunks)} chunks to Chroma...")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path,
            collection_name=collection_name
        )
        print(f"[INFO] Database created in {time.time() - start_time:.2f} seconds.")

    else:
        print(f"[INFO] Loading existing ChromaDB at: {db_path}")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=db_path,
            collection_name=collection_name,
        )

    # Information
    try:
        ids = vector_store.get(include=[])["ids"]
        print(f"Vector DB contains {len(ids)} chunks.")
    except Exception:
        print("Warning: unable to count stored documents.")

    return vector_store.as_retriever(search_kwargs={"k": 5})
