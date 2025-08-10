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

<img width="956" height="445" alt="7" src="https://github.com/user-attachments/assets/4e6eee61-f942-4628-90e7-8efdf0f73e11" />
<img width="959" height="443" alt="6" src="https://github.com/user-attachments/assets/b47bf2af-a306-43c0-818a-9b3de6211726" />
<img width="959" height="446" alt="5" src="https://github.com/user-attachments/assets/b87f2961-a5a2-4de4-b931-3fdcf2e64b23" />
<img width="954" height="443" alt="4" src="https://github.com/user-attachments/assets/be517321-e536-42b7-8f9b-4dc89aad4c41" />
<img width="956" height="446" alt="3" src="https://github.com/user-attachments/assets/bc90a66d-afd4-4cf9-b933-e3507535b852" />
<img width="959" height="444" alt="2" src="https://github.com/user-attachments/assets/4e7c362f-c464-403d-b9fe-ab395adb42a5" />
<img width="959" height="446" alt="1" src="https://github.com/user-attachments/assets/0ce896f0-1912-4dfd-a619-031d2da034c6" />
