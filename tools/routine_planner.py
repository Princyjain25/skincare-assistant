from langchain.tools import tool
from langchain.chains import RetrievalQA

class RoutinePlanner:
    def __init__(self, llm, retriever):
        """
        Initialize the routine planner with LLM and retriever.
        """
        self.routine_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff"
        )

    @tool
    def routine_planner(self, input: str) -> str:
        """
        Plan a skincare routine. Input should include skin type, concern, and budget.
        Example: 'oily skin with acne, low budget'.
        """
        prompt = f"Suggest a skincare routine for: {input}"
        return self.routine_chain.run(prompt)
