import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()
if __name__ == "__main__":
    print("This is the main retrieval logic.")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Without using embeddings vanila llm output provided below
    query = "What is Pinecone in Machine Learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke({})
    print(result.content)

    # Using Pinecone Vector Store and embeddings
    vector_store = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"),
        embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval_qa_chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retrieval_chain = vector_store.as_retriever(),
        combine_docs_chain = combine_docs_chain
    )
    result = retrieval_chain.invoke({"query": query})
    print(result)
