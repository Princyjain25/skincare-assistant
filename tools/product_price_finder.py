from langchain.tools import tool
from langchain.chains import RetrievalQA

class ProductPriceFinder:
    def __init__(self, llm, retriever):
        """
        Initialize the product finder tool with LLM and retriever.
        """
        self.product_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff"
        )

    @tool
    def suggest_products_by_price(self, input: str) -> str:
        """
        Suggest skincare products within a specific price range.
        Example input: 'budget under 1000 INR'
        """
        prompt = f"Suggest skincare products within the price range of {input}. Provide product names, brands, and prices."
        return self.product_chain.run(prompt)

    @tool
    def calculate_total_routine_price(self, routine: str) -> str:
        """
        Calculate the total price of all products in a suggested skincare routine.
        Example input: 'routine for dry skin'
        """
        prompt = f"List the products recommended for {routine}, along with their prices, and calculate the total cost."
        return self.product_chain.run(prompt)
