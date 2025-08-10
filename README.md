# HR-Policy-Assistant---AI-Chatbot
The HR Policy Assistant is an AI-powered chatbot designed to help employees navigate company policies and HR procedures efficiently. Built with Streamlit, LangChain, and Hugging Face models, this application provides instant, accurate answers to HR-related questions by referencing uploaded policy documents.

**Key Features**
- Natural Language Understanding: Interprets employee questions about policies, benefits, and procedures
- Document Retrieval: References specific policy documents and page numbers in responses
- Conversational Memory: Maintains context throughout discussions
- HR Admin Portal: Secure document upload and knowledge base updating
- Responsive UI: Clean, professional interface with streaming responses
- Citation System: Automatically references source documents for transparency

**Installation**

**Prerequisites**
- Python 3.9+
- pip package manager
- Hugging Face API token (free tier available)
- Optional: HR password for document uploads (set in environment variables)

**Setup Instructions**

**1. Clone the repository:**
- git clone https://github.com/CodeHub5199/HR-Policy-Assistant---AI-Chatbot
- cd hr-policy-assistant

**2. Create and activate a virtual environment:**
- python -m venv venv
- source venv/bin/activate  # Linux/Mac
- venv\Scripts\activate

**3. Install dependencies:**
- pip install -r requirements.txt

**4. Create a .env file with your configuration:**
- HF_TOKEN=your_huggingface_token
- HR_PASSWORD=your_secure_password  # Optional

**5. Run the application:**
- streamlit run new_app.py

**Usage**

**For Employees**
- Launch the application
- Ask questions in natural language (e.g., "How do I request vacation time?")
- Receive detailed responses with policy references

**For HR Administrators**
- Access the HR portal in the sidebar
- Enter the HR password
- Upload updated policy PDFs
- Click "Update Policy Database" to refresh the knowledge base

**Example Queries**
- What's our maternity leave policy?
- How many sick days do we get per year?
- What's the procedure for reporting harassment?
- Can you explain our health insurance benefits?
