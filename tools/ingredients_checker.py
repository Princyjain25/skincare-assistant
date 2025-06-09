# tools/ingredient_checker.py

from langchain.tools import tool
from langchain.chains import RetrievalQA

class IngredientChecker:
    def __init__(self, llm, retriever):
        """
        Initialize the ingredient checker with LLM and retriever.
        """
        self.ingredient_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff"
        )

    @tool
    def ingredient_checker(self, ingredient: str) -> str:
        """
        Check if an ingredient or product is safe or recommended for specific skin types or conditions.
        """
        query = f"Is '{ingredient}' suitable for sensitive or acne-prone skin? List any known issues."
        return self.ingredient_chain.run(query)
