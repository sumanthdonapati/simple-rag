import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# Initialize session state for chat history and vector store
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag_chain():
    model = ChatOllama(model="llama3.1:8b")
    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    
    chain = (
        RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
        | rag_prompt
        | model
        | StrOutputParser()
    )
    return chain

def process_url(url):
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        
        local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = InMemoryVectorStore(local_embeddings)
        vectorstore.add_documents(all_splits)
        return vectorstore, None
    except Exception as e:
        return None, str(e)

# Streamlit UI
st.title("Chat with Web Articles")

# URL input
url = st.text_input("Enter the URL of the article:")

if url and st.session_state.vectorstore is None:
    with st.spinner("Processing the article..."):
        vectorstore, error = process_url(url)
        if error:
            st.error(f"Error processing URL: {error}")
        else:
            st.session_state.vectorstore = vectorstore
            st.success("Article processed! You can now ask questions.")

# Initialize the chain
chain = initialize_rag_chain()

# Chat interface
if st.session_state.vectorstore is not None:
    # Display chat history
    for message in st.session_state.chat_history:
        role = "user" if message.startswith("Q: ") else "assistant"
        content = message.replace("Q: ", "").replace("A: ", "")
        st.chat_message(role).write(content)

    # Question input
    question = st.chat_input("Ask a question about the article")
    
    if question:
        st.chat_message("user").write(question)
        
        # Include previous responses in context for follow-up questions
        full_context = [msg for msg in st.session_state.chat_history] + [question]
        
        # Perform similarity search
        docs = st.session_state.vectorstore.similarity_search(" ".join(full_context))
        
        # Get response
        with st.chat_message("assistant"):
            response = chain.invoke({"context": docs, "question": question})
            st.write(response)
        
        # Store in chat history
        st.session_state.chat_history.append(f"Q: {question}")
        st.session_state.chat_history.append(f"A: {response}")

# Reset button
if st.button("Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.vectorstore = None
    st.rerun()