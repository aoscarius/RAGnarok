import os
import time
from datetime import date

# Remove pydantic warning
from pydantic import BaseModel
# Disables Pydantic warning about protected namespaces
BaseModel.model_config['protected_namespaces'] = () 

# Import langchain functions
from langchain_community.llms import LlamaCpp
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

        # Store config
        self.config = config

        # Chat history
        self.history = []
        # History window size
        self.history_k = config.get("llm", {}).get("history_window", 4)

        # Load LLM for Text Generation
        print(f"[INFO] Loading LLM model {os.path.basename(llm_path)}...")
        self.llm = LlamaCpp(
            model_path=llm_path,
            n_threads=max(1, os.cpu_count() // 2),
            n_gpu_layers=config.get("llm", {}).get("n_gpu_layers", 0),
            temperature=config.get("llm", {}).get("temperature", 0.1),
            top_p=config.get("llm", {}).get("top_p", 0.9),
            top_k=config.get("llm", {}).get("top_k", 40),
            max_tokens=config.get("llm", {}).get("max_tokens", 512),
            n_ctx=config.get("llm", {}).get("n_ctx", 4096),
            rope_freq_scale=config.get("llm", {}).get("rope_freq_scale", 1.0),
            n_batch=config.get("llm", {}).get("n_batch", 64),
            f16_kv=True,
            stop=config.get("llm", {}).get("stop", []),
            repeat_penalty=config.get("llm", {}).get("repeat_penalty", 1.15),
            streaming=config.get("llm", {}).get("streaming", True),
            use_mmap=True,
            use_mlock=False,
            echo=False,
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

    def promptrag(self, documents, question, history=""):
         # Load role templates from config
        ctrls = self.config.get("llm", {}).get("template", [])

        # Format history
        prompt_history = ""
        for chunk in history:
            rstart, rend = ctrls.get(chunk["role"], {}).get("start", ""), ctrls.get(chunk["role"], {}).get("end", "")
            prompt_history += f"{rstart}\n{chunk['content']}\n{rend}\n\n"

        # Build prompt template
        template = [
            # System role with instructions
            {
                "role": "system", 
                "content": f"""
                You are an expert technical assistant specialized in C, C++, embedded systems,
                and low-level software development.

                Answer the QUESTION using ONLY the DOCUMENTS and consider the HISTORY for context.
                If the DOCUMENTS do not contain enough information, reply exactly:
                "The provided documents do not contain enough information to answer this question."

                Rules:
                - Do not use external knowledge
                - If information is missing, say so explicitly
                - Do not invent information
                - Preserve identifiers exactly
                - Use Markdown
                """.strip()
            },
            # User role with context
            { 
                "role": "user", 
                "content": f"""
                HISTORY:
                {prompt_history}
                
                DOCUMENTS:
                {documents}

                QUESTION:
                {question}
                """.strip()
            }
        ]

        # Clean up leading spaces in contents
        prompt = ""
        for chunk in template:
            chunk["content"] = "\n".join(line.lstrip() for line in chunk["content"].splitlines())
            rstart, rend = ctrls.get(chunk["role"], {}).get("start", ""), ctrls.get(chunk["role"], {}).get("end", "")
            prompt += f"{rstart}\n{chunk['content']}\n{rend}\n\n"
        rstart = ctrls.get("assistant", {}).get("start", "")
        prompt += f"{rstart}\n"

        return prompt

    def stream(self, question):
        # Reset statistics
        self.token_stats = {
            "total_tokens": 0,
            "time_elapsed": 0.0,
            "tokens_per_second": 0.0
        }

        # Retrieval
        docs = self.embed.invoke(question)

        # Sanity check
        if not docs:
            yield "The provided documents do not contain enough information to answer this question."
            return
            
        # Print retrieved documents
        print(f"[INFO] Retrieved {len(docs)} documents for the question:")
        for i, d in enumerate(docs):
            print(f"[{i}] {d.metadata.get('source')} ({len(d.page_content)} chars)")

        # Update chat history
        rhistory = self.history[-self.history_k:]
        formatted_rhistory = ""
        for msg in rhistory:
            formatted_rhistory += f"{msg['role'].upper()}: {msg['content']}\n"

        # Format documents for prompt
        formatted_docs = "\n\n---\n\n".join(d.page_content for d in docs)

        # Stat timer for statistics       
        start_time = time.time()

        # Reset LLM internal state
        self.llm.client.reset()

        # Streaming and output (handles \n, \t, \r and unicode by default in Python 3)
        prompt = self.promptrag(formatted_docs, question)
        response = ""
        for tchunk in self.llm.stream(prompt):
        # for tchunk in self.ragChain.stream({"documents": formatted_docs, "question": question}):
            # Security check for GenerationChunks
            if isinstance(tchunk, GenerationChunk):
                token = tchunk.text
            else:
                token = str(tchunk)
            response += token
            self.token_stats["total_tokens"] += 1
            yield token.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r') # Little fixes

        # Store response in the history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

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