import os
import time

# Remove pydantic warning
from pydantic import BaseModel
# Disables Pydantic warning about protected namespaces
BaseModel.model_config['protected_namespaces'] = () 

# Import langchain functions
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs import GenerationChunk

# Import custom functions
from vector import documentRag
class llmRag:
    def __init__(self, llm_path, embed_path, config):

        # Extract paths from config
        docs_path = config.get("data", {}).get("docs_path", "docs")
        db_path = config.get("data", {}).get("db_path", "chromadb")

        if not os.path.isdir(docs_path):
            print(f"[ERROR] docs directory not found: {docs_path}. RAG will not work.")

        # Self contained statistics dictionary
        self.token_stats = {
            "total_tokens": 0,
            "time_elapsed": 0.0,
            "tokens_per_second": 0.0
        }

        # Load LLM for Text Generation
        print(f"[INFO] Loading LLM model {os.path.basename(llm_path)}...")
        self.llm = LlamaCpp(
            model_path=llm_path,
            n_threads=int(os.cpu_count() / 3),
            n_gpu_layers=config.get("llm", {}).get("n_gpu_layers", 0),
            temperature=config.get("llm", {}).get("temperature", 0.2),
            top_p=config.get("llm", {}).get("top_p", 0.9),
            max_tokens=config.get("llm", {}).get("max_tokens", 1024),
            n_ctx=config.get("llm", {}).get("n_ctx", 4096),
            n_batch=config.get("llm", {}).get("n_batch", 64),
            stop=[
                "<|user|>",
                "<|system|>",
                "<|assistant|>",
            ],
            streaming=config.get("llm", {}).get("streaming", True),
            verbose=config.get("llm", {}).get("verbose", False)
        )

        # Load Embedder for Retrieval
        print(f"[INFO] Loading Embed model {os.path.basename(embed_path)}...")
        self.embed = documentRag(
            source_directory=docs_path,
            db_path=db_path,
            model_path = embed_path,
            collection_name = config.get("data", {}).get("db_collection", "documents"),
            chunk_size = config.get("data", {}).get("chunk_size", 1000),
            chunk_overlap = config.get("data", {}).get("chunk_overlap", 150),
            config = config
        )

        # RAG Template
        self.system_istruct = """
You are an expert technical assistant. Use the documents to answer.
Your primary function is to answer the user's question **EXCLUSIVELY** based on the content of the provided 'Relevant documents' section.
Do not use any external knowledge. If the provided documents do not contain the answer, state clearly that the information is not available in the context.
        """
        self.template =  """
Relevant documents:
---
{documents}
---

Question:
{question}

Provide a clear, correct, concise, and helpful answer, strictly following the system instructions.
        """

        # Create RAG chain
        self.prompt = ChatPromptTemplate.from_template(self.system_istruct + self.template)
        self.ragChain = self.prompt | self.llm 

    def stream(self, question):
        # Reset statistics
        self.token_stats = {
            "total_tokens": 0,
            "time_elapsed": 0.0,
            "tokens_per_second": 0.0
        }

        # Retrieval
        docs = self.embed.invoke(question)
        formatted_docs = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "No relevant documents found."
        print(f"[INFO] Retrieved {len(docs)} documents for the question")
        input_data = {"documents": formatted_docs, "question": question}

        # Stat timer for statistics       
        start_time = time.time()

        # Streaming and output (handles \n, \t, \r and unicode by default in Python 3)
        for tchunk in self.ragChain.stream(input_data):
            # Security check for GenerationChunks
            if isinstance(tchunk, GenerationChunk):
                token = tchunk.text
            else:
                token = str(tchunk)
            self.token_stats["total_tokens"] += 1
            yield token.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r') # Little fixes

        # Stop timer and update statistics
        self.token_stats["time_elapsed"] = time.time() - start_time
        
        if self.token_stats["time_elapsed"] > 0 and self.token_stats["total_tokens"] > 0:
            self.token_stats["tokens_per_second"] = \
                self.token_stats["total_tokens"] / self.token_stats["time_elapsed"]
        else:
            self.token_stats["tokens_per_second"] = 0.0
    
    def getStats(self):
        if self.token_stats["time_elapsed"] > 0:
            if self.token_stats["tokens_per_second"] == 0.0 and self.token_stats["total_tokens"] > 0:
                 self.token_stats["tokens_per_second"] = \
                    self.token_stats["total_tokens"] / self.token_stats["time_elapsed"]
        
        return self.token_stats