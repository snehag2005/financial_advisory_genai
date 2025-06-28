from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Print out values to confirm .env is working
print("AZURE_OPENAI_API_KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
print("AZURE_OPENAI_ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("AZURE_OPENAI_DEPLOYMENT_NAME:", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
print("AZURE_COGNITIVE_SEARCH_ENDPOINT:", os.getenv("AZURE_COGNITIVE_SEARCH_ENDPOINT"))
print("AZURE_COGNITIVE_SEARCH_API_KEY:", os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY"))
print("AZURE_COGNITIVE_SEARCH_INDEX_NAME:", os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME"))
