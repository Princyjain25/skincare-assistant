import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.agents import Tool

from tools.ingredients_checker import IngredientChecker
from tools.routine_planner import RoutinePlanner
from tools.product_price_finder import ProductPriceFinder
from tools.currency_converter import CurrencyConverter 
from tools.search_tool import SkincareSearchTool

from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

# --- Configuration ---
# Fetch from env vars
load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "orca-mini:3b") 
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "skin_data")
RATE_GBP_TO_INR = float(os.getenv("RATE_GBP_TO_INR", "105.0"))


# Mapping of CSV filenames to their respective fields for merging
CSV_FIELD_MAP = {
    "skincare_routines.csv": ["skin_type", "description", "cleanse_step", "moisturize_step"],
    "db_summary.csv": ["summary", "context"],
    "model_skintypes.csv": ["skintype_name", "description"],
    "model_concerns.csv": ["concern_name", "symptoms", "related_conditions"],
    "model_conditions.csv": ["condition_name", "description", "treatments"],
    "model_categories.csv": ["category_name", "examples"],
    "model_interactions.csv": ["ingredient", "interaction", "risk"],
    "skincare_products.csv": ["product_name", "price", "brand", "category", "description"],
    "db_ingred_summary.csv": ["category_id", "category_name", "product_id", "num_ingredients", "date_updated"],
}


def load_and_merge_skin_data_as_documents(folder_path: str, delimiter: str = " | ") -> list[Document]:
    print(f"Loading documents from: {folder_path}")
    loaded_docs = []

    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                print(f"Loading: {filename}")

                fields_to_merge = CSV_FIELD_MAP.get(filename)
                if not fields_to_merge:
                    print(f"Skipping '{filename}': no field map defined.")
                    continue

                # loader = CSVLoader(file_path=file_path)
                loader = CSVLoader(file_path=file_path, encoding="utf-8")

                docs = loader.load()

                for doc in docs:
                    # Safely combine specified fields from metadata
                    merged_text = delimiter.join(
                        str(doc.metadata.get(field, "")).strip() for field in fields_to_merge
                    )
                    
                    # Add filename to metadata for traceability
                    metadata = doc.metadata.copy()
                    metadata["source_file"] = filename
                    
                    # Create new Document with merged text and preserved metadata
                    new_doc = Document(page_content=merged_text, metadata=metadata)
                    loaded_docs.append(new_doc)

        print(f"Loaded and processed {len(loaded_docs)} documents.")

        if not loaded_docs:
            print(f"No documents created from '{folder_path}'.")

    except Exception as e:
        print(f"Error loading documents from {folder_path}: {e}")
        loaded_docs = []

    return loaded_docs

print(f"Using Ollama model: {OLLAMA_MODEL}")
print("-" * 30)

# --- 1. Set up the LLM ---
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
print("LLM Initialized.")

# --- 2. Load skin data ---
docs = load_and_merge_skin_data_as_documents(DOCS_FOLDER)

# --- 3. Create embeddings ---
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

print("Embeddings Initialized.")

# --- 4. Create a vector store and retriever ---
if len(docs) == 0:
    raise ValueError("No documents found. Check your CSV loader path/format.")
try:
    
    vector_store = FAISS.from_documents(docs, embeddings)
    # Retrieve the top 3 relevant docs to provide more context
    # Reduces redundancy and increases diversity in results
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 20})
    print("Vector Store and Retriever Initialized using FAISS.")
    print(f"Total vectors stored: {len(vector_store.index_to_docstore_id)}")

except Exception as e:
    print(f"Error initializing FAISS.Error: {e}")
    exit()


# --- 5. Initialize tools ---
ingredient_tool = IngredientChecker(llm, retriever)
routine_tool = RoutinePlanner(llm, retriever)
price_tool = ProductPriceFinder(llm, retriever)
converter_tool = CurrencyConverter(RATE_GBP_TO_INR)
search_tool = SkincareSearchTool(llm, retriever)

# --- 6. Collect tool methods ---
tools = [
    Tool.from_function(
        func=ingredient_tool.ingredient_checker,
        name="ingredient_checker",
        description="Check safety and side effects of skincare ingredients"
    ),
    Tool.from_function(
        func=routine_tool.routine_planner,
        name="routine_planner",
        description="Plan skincare routines based on skin type, concern, and budget"
    ),
    Tool.from_function(
        func=price_tool.suggest_products_by_price,
        name="suggest_products_by_price",
        description="Suggest skincare products based on budget"
    ),
    Tool.from_function(
        func=price_tool.calculate_total_routine_price,
        name="calculate_total_routine_price",
        description="Calculate total price of a skincare routine"
    ),
    Tool.from_function(
        func=converter_tool.convert_gbp_to_inr,
        name="convert_gbp_to_inr",
        description="Convert GBP to INR"
    ),
    Tool.from_function(
        func=search_tool.search_skincare_db,
        name="search_skincare_db",
        description="Search general skincare database for ingredients, concerns, routines"
    )
]


custom_prompt_template = """
You are an expert AI assistant specializing in personalized skincare advice.

You have access to the following tools:
{tools}

You have been provided with the following context information retrieved from relevant documents:
You can use tools to check ingredients, suggest routines, recommend products by price, convert currency, and search skincare info.

If the context below doesn't answer the question, choose the best tool to help.
<RAG_CONTEXT_START>
{rag_context}
<RAG_CONTEXT_END>

Answer the following question. First, carefully review the RAG_CONTEXT If yes, respond directly based on it.
If not, or if the question involves external tasks (like pricing, ingredients, or conversion), use the appropriate tool.

Use this format exactly:

Question: the input question you must answer
Thought: Think step-by-step. Analyze if the RAG_CONTEXT answers the question. Even if yes, ALWAYS follow with 'Action: Final Answer' and the rest of the format.
Action: Choose one — either a tool name from [{tool_names}] OR "Final Answer"
Action Input: Provide either the tool input or "N/A" if Final Answer
Observation: Tool result or note that RAG_CONTEXT was sufficient
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I now have all the information needed to answer the user's question.
Final Answer: Provide a human-readable, helpful answer to the user's question.


**Guidelines:**
- Do NOT list tools. You must pick and use only ONE tool per Action step.
- Never describe tools unless you are using them via Action.
- Use `ingredient_checker` to assess ingredient safety.
- Use `routine_planner` to build routines based on skin type, concern, and budget.
- Use `suggest_products_by_price` to recommend budget-friendly products.
- Use `calculate_total_routine_price` to compute full routine costs.
- Use `convert_gbp_to_inr` to convert prices.
- Use `search_skincare_db` to retrieve general information on ingredients, routines, or skin conditions.
- If RAG_CONTEXT clearly answers the question, skip tools and go straight to the final answer.

Begin!

Question: {input}
RAG Context Provided:
{rag_context}
{agent_scratchpad}
"""

tool_names = [tool.name for tool in tools]

prompt = PromptTemplate(
    input_variables=["agent_scratchpad", "input", "tool_names", "tools", "rag_context"],
    template=custom_prompt_template,
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4,
    early_stopping_method="force"
)


print("💬 Skincare Assistant Ready!")
print("Type your skincare question below. Type 'quit' to exit.")

while True:
    try:
        user_input = input("\n🧑 You: ")
        if user_input.strip().lower() == 'quit':
            print("Skincare assistant signing off. Don’t forget your SPF!")
            break

        # Step 1: Retrieve relevant context using the vector retriever
        retrieved_docs = retriever.invoke(user_input)
        rag_context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])

        if not rag_context_str.strip():
            rag_context_str = "No relevant documents found for this query."

        print("\n🔍 Context Retrieved:")
        print(rag_context_str if rag_context_str.strip() else "No context available.")
        # Step 2: Run the agent executor with the user input and RAG context
        agent_input = {
            "input": user_input,
            "rag_context": rag_context_str,
        }

        print("\nAgent is thinking...")
        response = agent_executor.invoke(agent_input)

        # Step 3: Display final answer
        print("\nFinal Answer:")
        print(response['output'])

    except Exception as e:
        print(f"\nError occurred: {e}")

