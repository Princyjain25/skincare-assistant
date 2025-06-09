from langchain.tools import tool
from langchain.chains import RetrievalQA

class SkincareSearchTool:
    def __init__(self, llm, retriever):
        """
        Initialize the search tool with a language model and a retriever.
        """
        self.qa_tool = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff"
        )

    @tool
    def search_skincare_db(self, query: str) -> str:
        """
        Run a search query against the skincare database using the RetrievalQA chain.
        """
        return self.qa_tool.run(query)
