from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1:8b",
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.runnables import RunnablePassthrough

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chain = (
    RunnablePassthrough.assign(context=lambda input: format_docs(input["context"]))
    | rag_prompt
    | model
    | StrOutputParser()
)

question = "what is this article about? can you tell me in english"

docs = vectorstore.similarity_search(question)

# Initialize chat history
chat_history = []

chat_history = []

while True:
    question = input("\nEnter your question (or 'quit' to exit): ")
    
    if question.lower() == 'quit':
        break
        
    # Include previous responses in context for follow-up questions
    full_context = chat_history + [question]
    
    # Perform similarity search with current question and previous responses
    docs = vectorstore.similarity_search(" ".join(full_context))
    
    print('\n---------------------------------------')
    # Stream the response
    response_chunks = []
    for chunk in chain.stream({"context": docs, "question": question}):
        print(chunk, end="", flush=True)
        response_chunks.append(chunk)
    
    # Combine chunks to store in chat history
    response = "".join(response_chunks)
    
    # Store question and response in chat history
    chat_history.append(f"Q: {question}\nA: {response}")
    print('\n---------------------------------------')