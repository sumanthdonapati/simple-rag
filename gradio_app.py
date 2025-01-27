import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import chromadb

class RAGChatbot:
    def __init__(self):
        self.vectorstore = None
        self.chain = None
        self.chat_history = []
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=0
        )
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Initialize LLM
        self.model = ChatOllama(model="llama3.1:8b")
        
        # Define RAG prompt template
        self.RAG_TEMPLATE = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        <context>
        {context}
        </context>

        Question: {question}
        Previous conversation context:
        {chat_history}

        Answer:"""
        
        self.rag_prompt = ChatPromptTemplate.from_template(self.RAG_TEMPLATE)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_chat_history(self):
        return "\n".join(self.chat_history)

    def load_article(self, url):
        try:
            # Load and process the article
            loader = WebBaseLoader(url)
            data = loader.load()
            all_splits = self.text_splitter.split_documents(data)
            
            # Create new vectorstore
            if self.vectorstore:
                self.vectorstore = None
            
            self.vectorstore = Chroma.from_documents(
                documents=all_splits, 
                embedding=self.embeddings
            )
            
            # Initialize the chain
            self.chain = (
                RunnablePassthrough.assign(
                    context=lambda input: self.format_docs(input["context"]),
                    chat_history=lambda _: self.format_chat_history()
                )
                | self.rag_prompt
                | self.model
                | StrOutputParser()
            )
            
            # Reset chat history
            self.chat_history = []
            
            return "Article loaded successfully! You can now start asking questions."
        except Exception as e:
            return f"Error loading article: {str(e)}"

    def chat(self, message, history):
        try:
            if not self.vectorstore:
                return "Please load an article first!"
            
            # Get relevant documents
            docs = self.vectorstore.similarity_search(message)
            
            # Generate response
            response = self.chain.invoke({
                "context": docs,
                "question": message
            })
            
            # Update chat history
            self.chat_history.append(f"User: {message}")
            self.chat_history.append(f"Assistant: {response}")
            
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Create Gradio interface
def create_interface():
    chatbot = RAGChatbot()
    
    with gr.Blocks() as demo:
        gr.Markdown("# Chat with Any Article")
        
        with gr.Row():
            url_input = gr.Textbox(
                label="Enter article URL",
                placeholder="https://example.com/article"
            )
            load_button = gr.Button("Load Article")
        
        status_message = gr.Textbox(label="Status")
        
        chatbot_interface = gr.Chatbot(
            label="Chat History",
            height=400
        )
        
        msg_input = gr.Textbox(
            label="Your message",
            placeholder="Ask a question about the article...",
            lines=2
        )
        
        # Handle loading article
        load_button.click(
            fn=chatbot.load_article,
            inputs=[url_input],
            outputs=[status_message]
        )
        
        # Handle chat interaction
        msg_input.submit(
            fn=chatbot.chat,
            inputs=[msg_input, chatbot_interface],
            outputs=[chatbot_interface],
            show_progress=True
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)