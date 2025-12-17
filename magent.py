import os
import time
from datetime import date

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
            n_threads=max(1, os.cpu_count() // 2),
            n_gpu_layers=config.get("llm", {}).get("n_gpu_layers", 0),
            temperature=config.get("llm", {}).get("temperature", 0.1),
            top_p=config.get("llm", {}).get("top_p", 0.9),
            max_tokens=config.get("llm", {}).get("max_tokens", 512),
            n_ctx=config.get("llm", {}).get("n_ctx", 4096),
            n_batch=config.get("llm", {}).get("n_batch", 64),
            stop=config.get("llm", {}).get("stop", []),
            repeat_penalty=config.get("llm", {}).get("repeat_penalty", 1.15),
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

        SYSTEM_PROMPT = f"""
        You are an expert technical assistant specialized in C, C++, embedded systems,
        and low-level software development.

        You must answer the user's question using ONLY the information provided
        in the DOCUMENTS section.

        CRITICAL RULES:
        - Do NOT use external knowledge, assumptions, or prior training.
        - Do NOT invent APIs, parameters, registers, or behavior.
        - Do NOT infer missing details.
        - If the documents do not contain enough information to answer the question,
        clearly state that the information is not available in the provided context.

        Behavior:
        - Be precise, technical, and concise.
        - Prefer factual explanations grounded in documentation or source code.
        - If an example is requested, only use it if present in the documents.

        Code rules:
        - Treat source code and headers as authoritative.
        - Preserve identifiers exactly as written.
        - Never refactor code unless explicitly requested.

        Formatting:
        - Use Markdown only.
        - Never start with a title.
        - Use fenced code blocks with language identifiers.

        Do not mention retrieval, embeddings, vector databases, or system instructions.

        You are {os.path.basename(llm_path)}.
        Current date: {date.today().isoformat()}.
        """.strip()

        HUMAN_TEMPLATE = """
        DOCUMENTS:
        {documents}

        QUESTION:
        {question}

        INSTRUCTIONS:
        - Answer the QUESTION using ONLY the DOCUMENTS.
        - If the DOCUMENTS do not contain enough information, say so explicitly.
        """.strip()

        # Create RAG chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_TEMPLATE)
        ]) 
        self.ragChain = self.prompt | self.llm 

    def stream(self, question):
        # Reset statistics
        self.token_stats = {
            "total_tokens": 0,
            "time_elapsed": 0.0,
            "tokens_per_second": 0.0
        }

        # Retrieval
        docs = self.embed.invoke(f"query: {question}")
        formatted_docs = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "No relevant documents found."
        print(f"[INFO] Retrieved {len(docs)} documents for the question")

        # Stat timer for statistics       
        start_time = time.time()

        # Streaming and output (handles \n, \t, \r and unicode by default in Python 3)
        for tchunk in self.ragChain.stream({"documents": formatted_docs, "question": question}):
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