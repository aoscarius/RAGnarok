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

# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
# https://huggingface.co/NoelJacob/Meta-Llama-3-8B-Instruct-Q4_K_M-GGUF/resolve/main/meta-llama-3-8b-instruct.Q4_K_M.gguf

class llmRag:
    def __init__(self, llm_path, embed_path, docs_path, db_path):
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
            n_gpu_layers=0,
            temperature=0.2,
            top_p=0.9,
            max_tokens=1024,
            n_ctx=4096,
            n_batch=64,
            verbose=False
        )

        # Load Embedder for Retrieval
        print(f"[INFO] Loading Embed model {os.path.basename(embed_path)}...")
        self.embed = documentRag(
            source_directory=docs_path,
            db_path=db_path,
            model_path = embed_path
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
        end_time = time.time()
        self.token_stats["time_elapsed"] = end_time - start_time
        
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