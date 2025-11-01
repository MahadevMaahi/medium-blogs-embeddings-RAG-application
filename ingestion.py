import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Textloader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
if __name__ == "__main__":
    print("This is the ingestion.")

    loader = Textloader("./mediumblog1.txt", "utf8")
    document = loader.load()

    print("this is the Splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Number of documents: {len(texts)}")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name = os.environ.get("INDEX_NAME"))
    print("Ingestion complete.")