import os
import sys
import json
import argparse

# Import TUI/GUI system
from flask import Flask, render_template, request, Response, stream_with_context
from tui import CursesTUI

# Import custom functions
from utils import modelDownload
from magent import llmRag

ragAgent = None
app = Flask(__name__, template_folder=".", static_folder=".")

# --- TUI Mode Implementation ---
def tui_mode():
    """ Runs the TUI (Textual User Interface) mode. """
    global ragAgent # Ensure globals are accessible

    tui = CursesTUI(ragAgent=ragAgent, ragName="Ragnarok")
    tui.run()

# --- CLI Mode Implementation ---
def cli_mode():
    """ Runs the CLI (Command Line Interface) mode. """
    global ragAgent # Ensure globals are accessible

    print("\n-------------------------------------------")
    print("Ragnarok")
    print("-------------------------------------------")
    print("Welcome to Ragnarok Agent CLI. Could I help you? Ask me your question.")
    print()

    messages = 0
    while True:
        question = input("[You]:\n(q to quit): ").strip()

        # Check for question
        if question.lower() in ("q", "quit", "exit"):
            break
        if not question:
            continue
        
        print("\n[Ragnarok]:")
        for token in ragAgent.stream(question):
            print(token, end="", flush=True)

        stats = ragAgent.getStats()
        messages += 2
        print(f'\n\nPerfs: {stats["tokens_per_second"]:0.2f} t/s | Messages: {messages}\n')
                
    print("\n[INFO] Exiting CLI.")

# --- FLASK Mode Implementation ---
@app.route("/")
def index():
    """ Loads the main chat page. """
    return render_template(
        "index.html",
        ragName = "Ragnarok"
    )

@app.route("/api/chat", methods=["POST"])
def chat():
    """ Endpoint to handle the user's question and streaming response. """
    global ragAgent # Ensure globals are accessible

    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return "Question cannot be empty.", 400

    def generate():
        """Generator that yields tokens from the LangChain chain."""
        for token in ragAgent.stream(question):
            # In the web context, tokens are returned directly.
            yield token

    return Response(stream_with_context(generate()), mimetype="text/plain")

@app.route('/api/stats', methods=['GET'])
def get_agent_stats():
    """ Endpoint to return agent performance statistics. """
    # Assuming the agent instance is global or easily accessible
    stats = ragAgent.getStats()
    return {
        "tokens_per_second": stats["tokens_per_second"]
    }

if __name__ == "__main__":
    with open("config.json", "r") as f:

        # Load the config file
        config = json.load(f)
        
        # Extract paths from config with defaults
        models_path = config.get("models_path", "models")
        llm_url = config.get("llm", {}).get("url", "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf")
        embed_url = config.get("embed", {}).get("url", "https://huggingface.co/nesall/bge-small-en-v1.5-Q4_K_M-GGUF/resolve/main/bge-small-en-v1.5-q4_k_m.gguf")

        # Check for models already downloaded
        print(f"[INFO] Checking models into '{models_path}' folder...")
        llm_path = modelDownload(llm_url, models_path)
        embed_path = modelDownload(embed_url, models_path)

        # 1. Argument Parsing
        parser = argparse.ArgumentParser(description="RAG Agent with TUI, CLI and Web UI support.")
        parser.add_argument("--tui", action="store_true", help="Runs the chatbot in Textual User Interface (TUI) mode.")
        parser.add_argument("--cli", action="store_true", help="Runs the chatbot in Command Line Interface (CLI) mode.")
        args = parser.parse_args()
        
        # RAG System Initialization
        ragAgent = llmRag(llm_path, embed_path, config)

        if (len(sys.argv)-1) > 0:
            with open(os.devnull, 'w') as ferr:
                _stderr = sys.stderr
                sys.stderr = ferr

                if args.tui:
                    # TUI Mode
                    print("Starting TUI agent...")
                    tui_mode()
                elif args.cli:
                    # CLI Mode
                    print("Starting CLI agent...")
                    cli_mode()
                else:
                    print("Wrong argument")

                sys.stderr = _stderr
        else:
                # Web UI Mode (Flask)
                print("Starting WebUI agent...")
                app.run(host="0.0.0.0", port=5000)

