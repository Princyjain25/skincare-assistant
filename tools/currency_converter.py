from langchain.tools import tool

class CurrencyConverter:
    def __init__(self, rate_gbp_to_inr: float = 105.0):
        """
        Initialize with a default GBP to INR rate.
        You can update this rate manually or later use an API.
        """
        self.gbp_to_inr = rate_gbp_to_inr

    @tool
    def convert_gbp_to_inr(self, amount_str: str) -> str:
        """
        Convert GBP price to INR.
        Example input: '£87.00'
        """
        try:
            amount = float(amount_str.replace("£", "").strip())
            converted = amount * self.gbp_to_inr
            return f"£{amount:.2f} is approximately ₹{converted:.2f}"
        except Exception as e:
            return f"Error: Could not convert amount. Make sure input is like '£87.00'. Details: {e}"
