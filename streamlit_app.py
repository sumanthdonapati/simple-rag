import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Constants
RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

# Initialize components
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
model = ChatOllama(model="deepseek-r1")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit app
def main():
    st.set_page_config(page_title="Chat with Articles", layout="wide")
    st.title("📄 Chat with Articles")

    # Session state for storing embeddings, vector store, and chat history
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input: URL of the article
    url = st.text_input("Enter the URL of a blog or article:", "")
    if st.button("Load Document") and url:
        try:
            # Load document
            loader = WebBaseLoader(url)
            data = loader.load()

            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            all_splits = text_splitter.split_documents(data)

            # Generate embeddings
            embeddings = OllamaEmbeddings(model="nomic-embed-text")

            # Create vector store with persistence
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits, 
                embedding=embeddings, 
                persist_directory="./chroma_db"  # Specify persistence directory
            )
            st.session_state.chat_history = []  # Reset chat history

            st.success("Document loaded and processed successfully!")
        except Exception as e:
            st.error(f"Failed to load the document: {e}")

    # Chat interface
    if st.session_state.vectorstore:
        st.subheader("Chat with the Document")

        # User input for the question
        question = st.text_input("Ask a question:")
        if st.button("Submit") and question:
            try:
                # Retrieve relevant documents
                docs = st.session_state.vectorstore.similarity_search(question)

                # Create RAG chain
                chain = (
                    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
                    | rag_prompt
                    | model
                    | StrOutputParser()
                )

                # Generate response
                response_chunks = []
                for chunk in chain.stream({"context": docs, "question": question}):
                    response_chunks.append(chunk)
                response = "".join(response_chunks)

                # Update chat history
                st.session_state.chat_history.append({"question": question, "response": response})
            except Exception as e:
                st.error(f"Failed to process your question: {e}")

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for chat in st.session_state.chat_history:
                st.markdown(f"**Q:** {chat['question']}\n**A:** {chat['response']}")

if __name__ == "__main__":
    main()
