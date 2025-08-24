import os
import time
import re
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
THANKS = ["thank you", "thanks", "appreciate it"]
UPLOAD_DIR = "./uploads"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_CONFIG = {
    "base_url": "https://router.huggingface.co/v1",
    "model": "openai/gpt-oss-20b:groq"
}
BATCH_SIZE = 100
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# UI Configuration
PAGE_CONFIG = {
    "page_title": "HR Policy Bot",
    "page_icon": "🏢",
    "layout": "centered",
    "initial_sidebar_state": "expanded"
}

COLORS = {
    "primary": "#4B7BFF",
    "secondary": "#6C757D",
    "success": "#28A745",
    "danger": "#DC3545",
    "light": "#F8F9FA",
    "dark": "#343A40"
}

def configure_page() -> None:
    """Configure Streamlit page settings with enhanced styling."""
    st.set_page_config(
        **PAGE_CONFIG,
        menu_items={
            'About': """
            ### HR Policy Assistant - Technical Details
            
            **AI Model:**  
            `openai/gpt-oss-20b:groq`  
            *Powered by Hugging Face API*
            
            **Embedding Model:**  
            `all-MiniLM-L6-v2`  
            *Sentence Transformers*

            **Vector Database:**  
            `ChromaDB`  
            *Vector Store*
            
            **Version:**  
            1.0.0
            
            ---
            For support, please contact IT Department
            """
        }
    )
    
    # Enhanced CSS styling
    st.markdown(f"""
        <style>
            /* Main content area */
            .main {{
                background-color: {COLORS['light']};
                padding: 2rem;
                border-radius: 10px;
            }}
            
            /* Chat containers */
            
            
            .stChatMessage {{
                width: 85%;
                margin-bottom: 1rem;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            
            /* User message specific */
            .stChatMessage.user {{
                background-color: {COLORS['primary']}15;
                border-left: 4px solid {COLORS['primary']};
            }}
            
            /* Assistant message specific */
            .stChatMessage.assistant {{
                background-color: white;
                border-left: 4px solid {COLORS['secondary']};
            }}
            
            /* Sidebar styling */
            .sidebar .sidebar-content {{
                padding: 2rem 1rem;
                background-color: {COLORS['light']};
            }}
            
            /* Divider styling */
            hr {{
                margin: 1.5rem 0;
                border: 0;
                height: 1px;
                background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.75), rgba(0,0,0,0));
            }}
            
            /* Citations and references */
            .citation {{
                color: {COLORS['secondary']}; 
                font-size: 0.85em; 
                margin-top: 0.75rem;
                padding-top: 0.5rem;
                border-top: 1px dashed {COLORS['secondary']}50;
            }}
            
            /* Policy steps styling */
            .policy-steps {{
                margin-left: 1.5rem;
                padding-left: 1rem;
                border-left: 2px solid {COLORS['primary']}50;
            }}
            
            .policy-step {{
                margin-bottom: 0.75rem;
                padding: 0.25rem 0;
            }}
            
            /* Buttons */
            .stButton>button {{
                background-color: {COLORS['primary']};
                color: white;
                border-radius: 6px;
                border: none;
                padding: 0.5rem 1rem;
                transition: all 0.3s;
            }}
            
            .stButton>button:hover {{
                background-color: {COLORS['primary']}DD;
                transform: translateY(-1px);
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            
            /* Input fields */
            .stTextInput>div>div>input {{
                border-radius: 6px !important;
                border: 1px solid {COLORS['secondary']}30 !important;
            }}
            
            /* Status messages */
            .stAlert {{
                border-radius: 6px;
            }}
            
            /* Spinner animation */
            .stSpinner>div {{
                margin: 0 auto;
            }}
        </style>
    """, unsafe_allow_html=True)

def initialize_session_state() -> None:
    """Initialize or reset session state variables."""
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "processing_files": False,
        "last_upload_time": None,
        "conversation_memory": [],
        "first_run": True
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def format_response_with_steps(response_text: str) -> str:
    """
    Format the response text to ensure proper numbering and bullet points.
    """
    # Convert numbered lists to proper markdown format
    response_text = re.sub(r'(\d+)\.\s', r'\1. ', response_text)
    
    # Ensure steps are properly formatted
    if "steps:" in response_text.lower() or "step " in response_text.lower():
        response_text = re.sub(r'(?i)(steps?:?\s*)', '\n', response_text)
        response_text = re.sub(r'(\d+\.)\s+', r'\1 ', response_text)
    
    # Convert any remaining bullet points to markdown
    response_text = re.sub(r'[\•\*\-]\s+', '- ', response_text)
    
    return response_text

def extract_and_format_sources(source_docs: List[Document]) -> List[Dict[str, str]]:
    """
    Extract and format source information from documents.
    """
    unique_sources = {}
    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", "N/A")
        key = f"{source}|{page}"
        if key not in unique_sources:
            unique_sources[key] = {
                "source": source,
                "page": page
            }
    return list(unique_sources.values())

def is_greeting(text: str) -> bool:
    """Check if the input text is a greeting."""
    return any(greet in text.lower() for greet in GREETINGS)

def is_thanks(text: str) -> bool:
    """Check if the input text is an expression of thanks."""
    return any(thanks in text.lower() for thanks in THANKS)

def get_greeting_response() -> Tuple[str, List[Document]]:
    """Return a greeting response."""
    return "Hello! I'm your HR Policy Assistant. How can I help you today?", []

def get_thanks_response() -> Tuple[str, List[Document]]:
    """Return a thanks response."""
    return "You're welcome! Let me know if you have any other questions about company policies.", []

def load_and_split_pdfs(pdf_paths: List[str]) -> List[Document]:
    """
    Load and split PDF documents into chunks with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    docs = []
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = os.path.basename(path)
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1
            docs.extend(loaded_docs)
        except Exception as e:
            st.error(f"Error loading {path}: {str(e)}")
            continue
            
    return text_splitter.split_documents(docs)

def initialize_vector_store() -> Chroma:
    """Initialize or get existing Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

def get_vector_store() -> Chroma:
    """Get the current vector store from session state or initialize a new one."""
    if st.session_state.vector_store is None:
        st.session_state.vector_store = initialize_vector_store()
    return st.session_state.vector_store

def add_to_vector_store(pdf_paths: List[str]) -> None:
    """
    Process and add documents to the vector store in batches.
    """
    try:
        chunks = load_and_split_pdfs(pdf_paths)
        vector_store = get_vector_store()
        
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            if batch_chunks:
                vector_store.add_documents(batch_chunks)
                
        st.session_state.last_upload_time = time.time()
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        raise

def get_contextualized_qa_chain(hf_token) -> Any:
    """
    Create a QA chain that's aware of conversation history.
    """
    try:
        llm = ChatOpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=hf_token,
            model=LLM_CONFIG["model"],
            temperature=0
        )
        
        # System prompt with strict formatting instructions
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an HR policy assistant. Answer questions using ONLY the provided context.
            
            FORMATTING RULES:
            1. Always present steps or procedures as a numbered list
            2. Each step should be concise and begin with a verb
            3. Never include "Step X:" - just use numbers
            4. Leave exactly one blank line before lists
            5. Never write "Here are the steps" - just present the steps
            
            EXAMPLE:
            To initiate your resignation:

            1. Notify HR or your manager
            2. Provide at least three months notice
            3. Schedule an exit meeting
            4. Return company equipment
            
            Context:
            {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        return create_retrieval_chain(retriever, question_answer_chain)
        
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        raise

def process_user_query(query: str, qa_chain: Any) -> Tuple[str, List[Document]]:
    """
    Process a user query and generate an appropriate response with citations.
    """
    if is_greeting(query):
        return get_greeting_response()
    if is_thanks(query):
        return get_thanks_response()
    
    try:
        response = qa_chain.invoke({
            "input": query,
            "chat_history": st.session_state.chat_history
        })
        
        answer = response["answer"]
        source_docs = response.get("context", [])
        
        # Apply additional formatting to ensure consistent numbering
        answer = format_response_with_steps(answer)
        
        return answer, source_docs
        
    except Exception as e:
        return f"Sorry, I encountered an error processing your request: {str(e)}", []

def display_chat_history() -> None:
    """Display the chat history in the Streamlit UI."""
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(message.content)
            if hasattr(message, 'metadata') and message.metadata.get('sources'):
                sources = message.metadata['sources']
                st.markdown('<div class="citation"><strong>Reference:</strong></div>', 
                          unsafe_allow_html=True)
                for source in sources:
                    st.markdown(f'<div class="citation">📄 {source["source"]} (Page {source["page"]})</div>', 
                              unsafe_allow_html=True)

def display_streaming_response(answer: str, sources: List[Document]) -> None:
    """
    Display the assistant's response with a streaming effect and citations.
    """
    response_container = st.empty()
    full_response = ""
    
    # Split into paragraphs for better streaming
    paragraphs = answer.split('\n\n')
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        # Handle numbered lists specially
        if re.match(r'^\d+\.\s', para):
            with st.container():
                st.markdown('<div class="policy-steps">', unsafe_allow_html=True)
                for line in para.split('\n'):
                    if line.strip():
                        st.markdown(f'<div class="policy-step">{line}</div>', 
                                  unsafe_allow_html=True)
                        time.sleep(0.1)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            for word in para.split():
                full_response += word + " "
                response_container.markdown(full_response + "▌")
                time.sleep(0.02)
            full_response += "\n\n"
            response_container.markdown(full_response)
    
    # Display citations if available
    if sources:
        formatted_sources = extract_and_format_sources(sources)
        st.markdown('<div class="citation"><strong>Reference:</strong></div>', 
                   unsafe_allow_html=True)
        for source in formatted_sources:
            st.markdown(f'<div class="citation">📄 {source["source"]} (Page {source["page"]})</div>', 
                      unsafe_allow_html=True)

def is_hr() -> bool:
    """Check if the user has provided the correct HR password."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### HR Access Portal")
        hr_password = st.text_input(
            'Enter HR Password', 
            type='password',
            help="Enter the HR password to upload documents",
            key="hr_password_input"
        )
        return hr_password == os.getenv("HR_PASSWORD")

def handle_file_upload() -> None:
    """Handle the HR document upload process with enhanced UI."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### HR Document Management")
        st.markdown("Upload updated policy documents below.")
        
        new_files = st.file_uploader(
            'Select Policy PDFs',
            type='pdf',
            accept_multiple_files=True,
            help="Upload multiple PDF policy documents",
            label_visibility="collapsed"
        )
        
        if new_files:
            st.markdown(f"<small>Selected files: {', '.join([f.name for f in new_files])}</small>", 
                       unsafe_allow_html=True)
            
            if st.button('Update Policy Database', 
                        disabled=st.session_state.processing_files,
                        help="Click to process and update the policy knowledge base"):
                st.session_state.processing_files = True
                try:
                    pdf_paths = []
                    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Save files
                    status_text.info("Saving uploaded files...")
                    for i, uploaded_file in enumerate(new_files):
                        file_path = Path(UPLOAD_DIR) / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        pdf_paths.append(str(file_path))
                        progress_bar.progress((i + 1) / (len(new_files) * 2))
                    
                    # Process files
                    status_text.info("Processing documents...")
                    add_to_vector_store(pdf_paths)
                    progress_bar.progress(1.0)
                    
                    st.success("✅ Policy database updated successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error updating policies: {str(e)}")
                finally:
                    st.session_state.processing_files = False
                    progress_bar.empty()
                    status_text.empty()

def display_welcome_message() -> None:
    """Display a welcome message on first run."""
    if st.session_state.first_run:
        with st.chat_message("assistant"):
            st.write("Hello! I'm your HR Policy Assistant. I can help you with:")
            st.markdown("""
            - Company policies and procedures
            - Employee benefits information
            - HR processes and guidelines
            - Workplace regulations
            
            Ask me anything like:
            - "What's our vacation policy?"
            - "How do I request parental leave?"
            - "What's the dress code policy?"
            """)
        
        ai_message = AIMessage(
            content="Hello! I'm your HR Policy Assistant. How can I help you today?",
            metadata={"sources": []}
        )
        st.session_state.chat_history.append(ai_message)
        st.session_state.first_run = False

def main() -> None:
    """Main application function with enhanced UI."""
    configure_page()
    hf_token = st.sidebar.text_input('Enter HuggingFace Token', type="password", help='Enter HuggingFace Token to initiate chatbot')
    if hf_token:
        initialize_session_state()
        
        # Initialize QA chain with conversation memory
        qa_chain = get_contextualized_qa_chain(hf_token)
        
        # Application header with improved layout
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.title('HR Policy Assistant 🏢')
            st.markdown("""
                <div style="color: #6C757D; margin-bottom: 1.5rem;">
                    Get instant answers about company policies, benefits, and procedures.
                    For HR personnel, please log in via the sidebar.
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/477/477103.png", width=80)
        
        st.divider()
        
        # Display welcome message on first run
        display_welcome_message()
        
        # Display chat history
        display_chat_history()
        
        # Handle user input
        if query := st.chat_input("Ask about company policies...", key="user_query"):
            # Add user message to chat history
            user_message = HumanMessage(content=query)
            st.session_state.chat_history.append(user_message)
            
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                answer, source_docs = process_user_query(query, qa_chain)
                display_streaming_response(answer, source_docs)
            
            # Add assistant response to chat history with source metadata
            sources_metadata = extract_and_format_sources(source_docs)
            
            ai_message = AIMessage(
                content=answer,
                metadata={"sources": sources_metadata}
            )
            st.session_state.chat_history.append(ai_message)
        
        # HR document upload functionality
        if is_hr():
            handle_file_upload()
        else:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### HR Access")
            st.sidebar.info("HR personnel can log in to upload and update policy documents.")
    else:
        st.warning('Please enter HuggingFace Token to initiate chatbot')

if __name__ == "__main__":
    main()