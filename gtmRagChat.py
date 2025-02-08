from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
import os

class ChatSystem:
    def __init__(self):
        self.BASE_MEMORY_DIR = os.path.join(os.getcwd(), 'memory')
        self.ARTICLES_DIR = os.path.join(self.BASE_MEMORY_DIR, 'gtm/articles')
        self.conversation_history = []
        
        # Load and process PDFs
        self.documents = self.load_documents()
        self.chunks = self.split_documents(self.documents)
        self.embeddings = self.get_embeddings()
        
        # Create vector store
        self.vectordb = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.BASE_MEMORY_DIR, "chromadb")
        )

    def load_documents(self):
        """Load all PDFs from the articles directory"""
        loader = PyPDFDirectoryLoader(self.ARTICLES_DIR)
        return loader.load()

    def split_documents(self, documents):
        """Split documents into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        return text_splitter.split_documents(documents)

    def get_embeddings(self):
        """Get embeddings model"""
        return OllamaEmbeddings(model="nomic-embed-text")

    def create_message(self, message: str, role: str) -> dict:
        return {'role': role, 'content': message}

    def search_documents(self, query: str, k=9):
        """Search for relevant document chunks"""
        results = self.vectordb.similarity_search(query, k=k)
        context = ""
        sources = []
        
        for doc in results:
            # Extract just the filename without the path
            source = os.path.basename(doc.metadata.get('source', ''))
            if source not in sources:
                sources.append(source)
            context += f"\nFrom {source}:\n{doc.page_content}\n"
        
        return context, sources

    def process_query(self, query: str) -> str:
        """Process user query and maintain conversation history"""
        if query.lower() == "quit":
            return "Goodbye!"

        # Search for relevant context and get sources
        context, sources = self.search_documents(query)
        
        # Build messages array with system context and conversation history
        messages = [
            self.create_message(
                "You are a helpful GTM (Go-to-Market) strategy assistant. Use the provided context to answer questions accurately. "
                "If you're not sure about something, admit it rather than making things up. "
                "Keep responses concise and focused on GTM strategy. "
                "Always end your response with a list of sources used, formatted as 'Sources: [file1.pdf, file2.pdf, ...]'",
                'system'
            ),
            self.create_message(
                f"Here is the relevant context from the GTM documents:\n\n{context}\n\n"
                f"Remember to cite these sources at the end of your response: {sources}",
                'system'
            )
        ]
        
        # Add recent conversation history
        messages.extend(self.conversation_history[-5:])
        messages.append(self.create_message(query, 'user'))
        
        # Get response
        response = ollama.chat(model='mistral', messages=messages)
        response_content = response['message']['content']
        
        # If response doesn't include sources, add them
        if not response_content.strip().endswith(']'):
            response_content += f"\n\nSources: {sources}"
        
        # Update conversation history
        self.conversation_history.append(self.create_message(query, 'user'))
        self.conversation_history.append(self.create_message(response_content, 'assistant'))
        
        return response_content

def start_chat():
    chat_system = ChatSystem()
    print("\nWelcome to the GTM Strategy Assistant!")
    print("I can help answer questions about Go-to-Market strategy using our document library.")
    print("Type 'quit' to exit")
    print("\nWhat would you like to know about GTM strategy?\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
            
            response = chat_system.process_query(user_input)
            print("\n" + response + "\n")
            
            if user_input.lower() == "quit":
                break
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    start_chat()

