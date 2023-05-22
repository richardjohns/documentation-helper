import os
import time
from retrying import retry
from dotenv import load_dotenv
from glob import glob

from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone

# Load .env
load_dotenv()

# Set environment variables
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['PINECONE_ENVIRONMENT_REGION'] = os.getenv('PINECONE_ENVIRONMENT_REGION')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)
INDEX_NAME = "personal-knowledgebase"

# Define a retry decorator for network operations
@retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def load_and_split_file(file_path):
    # Load HTML file
    loader = BSHTMLLoader(file_path)
    raw_documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(raw_documents)

def ingest_docs(folder_path):
    # List to hold all documents
    documents = []

    # Convert the relative path to an absolute path
    absolute_folder_path = os.path.abspath(folder_path)

    # Check if the directory exists
    if not os.path.isdir(absolute_folder_path):
        print(f"No directory found at: {absolute_folder_path}")
        return

    # Log the absolute path
    print(f"Loading documents from: {absolute_folder_path}")

    # Get list of all HTML files in the specified folder and its subfolders
    html_files = glob(f"{folder_path}/**/*.html", recursive=True)
    print(f"Found {len(html_files)} HTML files.")

    for file_path in html_files:
        try:
            chunked_documents = load_and_split_file(file_path)

            # Update source URL
            for doc in chunked_documents:
                new_url = doc.metadata["source"]
                new_url = new_url.replace(folder_path, "https:/")
                doc.metadata.update({"source": new_url})

            # Add chunked documents to the main documents list
            documents.extend(chunked_documents)

        except Exception as e:
            print(f"Failed to load and split file {file_path}: {e}")

        # Throttling: wait before making the next request
        time.sleep(1)

    embeddings = OpenAIEmbeddings()
    print(f"Going to add {len(documents)} to Pinecone")
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorestore done ***")

if __name__ == "__main__":
    ingest_docs("langchain-docs")