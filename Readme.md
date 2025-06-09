# Simple Ollama RAG + Tools Example

This application demonstrates a basic example of using a locally running Ollama Large Language Model (LLM) combined with Retrieval-Augmented Generation (RAG) and custom Tools, orchestrated by the LangChain framework.

## Overview

The application allows you to interact with an AI agent that can:
1.  **Answer questions based on provided documents:** It uses RAG to search through a small, predefined set of documents about "Project Alpha" and "Project Beta".
2.  **Use custom tools:** It has access to tools that can retrieve the current date, current time, and calculate the number of days until a future date.
3.  **Leverage a local LLM:** All language processing is done locally using Ollama and a specified model (e.g., `llama3:8b`).

## Features

*   **Local LLM:** Uses Ollama for local, private LLM inference.
*   **RAG:** Implements basic RAG using in-memory FAISS vector store. Document context is retrieved *before* the agent is invoked and passed to it.
*   **Tools:** Demonstrates integrating custom Python functions as tools for the agent.
*   **Agent:** Utilizes a LangChain ReAct (Reasoning and Acting) agent. The agent is provided with RAG context and decides whether to use that context, its own knowledge, or one of the available tools to answer the user's query.
*   **LangChain:** Built using the LangChain framework for component integration.

## Setup


Follow these steps to set up and run the application:

1.  **Install Ollama:**
    If you haven't already, download and install Ollama from https://ollama.com/.

2.  **Pull an Ollama Model:**
    Open your terminal and pull the desired LLM. The script is configured for `llama3:8b` by default, but you can change the `OLLAMA_MODEL` variable in the script.
    ```bash
    ollama pull llama3:8b
    ```
    *(Optional: You can use other models like `llama3`, `mistral`, etc. Just make sure to update the `OLLAMA_MODEL` variable in `simple_rag_tools_app.py`)*

3.  **Clone or Download the Code:**
    Clone or Download - https://github.com/saurav-samantray/ollama-llm-rag-tools-example.git
    Ensure you have the `simple_rag_tools_app.py` file in your workspace (e.g., `c:\workspace\ai-agent\ollama-llm-rag-tools-example\`).
4.  **Create a Python Environment (Recommended):**
    It's good practice to use a virtual environment.
    ```bash
    cd c:\workspace\ai-agent\ollama-llm-rag-tools-example
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate # On Linux/macOS
    ```

5.  **Install Python Dependencies:**
    Install the required libraries using pip.
    ```bash
    pip install -r requirements.txt
    ```
    *   `langchain`, `langchain_community`, `langchain_ollama`, `langchainhub`: Core LangChain components and Ollama integration.
    *   `faiss-cpu`: Vector store library for RAG (CPU version).
    *   `python-dotenv`: Optional, for loading environment variables if needed later.

## Running the Application

1.  **Start Ollama (if not already running):**
    In *one* terminal, make sure the Ollama service is running. You can start it by running a model, which keeps the service active:
    ```bash
    ollama run llama3:8b
    ```
    Keep this terminal open.

2.  **Run the Python Script:**
    In *another* terminal (where you activated your virtual environment), navigate to the project directory and run the script:
    ```bash
    cd c:\workspace\ai-agent\ollama-llm-rag-tools-example
    python simple_rag_tools_app.py
    ```

3.  **Interact:**
    The script will initialize the components and prompt you for input. Try asking questions like:
    *   `What is Project Alpha about?`
    *   `Tell me about Project Beta funding.`
    *   `What is the date today?`
    *   `What time is it?`
    *   `When is the next meeting for Project Alpha and how many days are left until then?`
    *   `Can you tell me about Project Alpha and also today's date?`
    *   Type `quit` to exit.

    Observe the output with `verbose=True` to see the agent's thought process. Note how it first considers the RAG context provided to it, then decides if a tool is needed, processes the tool's input/output, and finally generates the response.

## Next Steps & Exploration

*   **More Documents:** Replace the simple `docs` list with loading documents from files (PDF, TXT, CSV) using LangChain's `DocumentLoader` classes.
*   **Document Chunking Strategies:** Experiment with different text splitters and chunk sizes for more effective RAG.
*   **Persistent Vector Store:** Instead of the in-memory FAISS, use a persistent vector store like ChromaDB or a cloud-based one.
*   **More Sophisticated Tools:** Create tools that interact with APIs (weather, stocks, databases) or perform more complex calculations.
*   **Different Agent Types:** Explore other LangChain agents (e.g., OpenAI Functions Agent if using OpenAI models, Conversational Agents).
*   **Error Handling:** Add more robust error handling around tool execution and LLM calls.
*   **Streaming:** Implement response streaming for a more interactive feel.
*   **UI:** Build a simple web interface using Streamlit or Flask.