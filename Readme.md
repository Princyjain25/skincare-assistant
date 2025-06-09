# Simple Skin Care Assistant (Ollama RAG + Tools Example)

This application demonstrates a basic AI-powered skin care assistant using a locally running Ollama Large Language Model (LLM), Retrieval-Augmented Generation (RAG), and custom tools, all orchestrated by the LangChain framework.

## Overview

The assistant can:
1.  **Answer skin care questions:** Uses RAG to search through a small set of skin care documents (e.g., routines, ingredients, tips).
2.  **Use helpful tools:** Provides the current date, time, and calculates days until a future appointment or event.
3.  **Run locally:** All AI processing is private and local using Ollama and a specified model (e.g., `llama3:8b`).

## Features

*   **Local LLM:** Private, local inference with Ollama.
*   **RAG:** Finds relevant info from your skin care documents before answering.
*   **Tools:** Includes date, time, and countdown helpers.
*   **Agent:** Uses LangChain's ReAct agent to combine knowledge, documents, and tools.
*   **Simple Python app:** Easy to run and modify.

## Setup

1.  **Install Ollama:**  
    Download and install from https://ollama.com/.

2.  **Pull a Model:**  
    ```bash
    ollama pull llama3:8b
    ```
    *(You can use other models if you prefer.)*

3.  **Get the Code:**  
    Clone or download this repo and ensure `simple_rag_tools_app.py` is in your workspace.

4.  **Create a Python Environment (Recommended):**  
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate # On Linux/macOS
    ```

5.  **Install Dependencies:**  
    ```bash
    pip install -r requirements.txt
    ```
    *(Includes LangChain, Ollama integration, FAISS, etc.)*

## Running the Assistant

1.  **Start Ollama:**  
    ```bash
    ollama run llama3:8b
    ```
    Keep this terminal open.

2.  **Run the Script:**  
    In another terminal:
    ```bash
    python simple_rag_tools_app.py
    ```

3.  **Ask Questions:**  
    Try:
    *   `What is a good morning skin care routine?`
    *   `Tell me about niacinamide benefits.`
    *   `What is the date today?`
    *   `How many days until my next dermatologist appointment on July 15?`
    *   `quit` to exit.

    The assistant will use your skin care documents and tools to answer.

## Next Steps

*   **Add More Documents:** Load your own skin care guides, ingredient lists, or FAQs.
*   **Customize Tools:** Add reminders, product lookup, or appointment scheduling.
*   **Build a UI:** Try Streamlit or Flask for a simple web interface.

