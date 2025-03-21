
# import os
# import faiss
# import google.generativeai as genai
# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain_core.prompts import PromptTemplate 
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from io import BytesIO
# import speech_recognition as sr
# from streamlit_mic_recorder import mic_recorder

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Existing PDF processing functions (unchanged)
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_stream = BytesIO(pdf.getvalue())
#         pdf_reader = PdfReader(pdf_stream)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. Structure your response with key points listed as bullet points starting with "- ". Do not use bold formatting (**text**) in your response; keep it plain text. Provide all relevant details, and if the answer is not in the context, say, "answer is not available in the context". Do not provide incorrect information.\n\n
#     Context:\n{context}\n
#     Question:\n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]

# def format_response(response):
#     lines = response.split("\n")
#     formatted_response = ""
#     for line in lines:
#         if line.strip().startswith("- "):
#             parts = line.strip()[2:].split(":", 1)
#             if len(parts) == 2:
#                 keyword = parts[0].strip()
#                 description = parts[1].strip()
#                 formatted_response += f"- <b>{keyword}</b>: {description}\n"
#             else:
#                 formatted_response += f"{line}\n"
#         else:
#             formatted_response += f"{line}\n"
#     return formatted_response

# # Function to transcribe audio from mic_recorder output
# def transcribe_audio(audio_bytes):
#     recognizer = sr.Recognizer()
#     audio_file = BytesIO(audio_bytes)
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Sorry, I couldn't understand the audio."
#         except sr.RequestError as e:
#             return f"Error with speech recognition: {str(e)}"

# def main():
#     st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")
#     st.markdown("""
#         <style>
#         .main-title { font-size: 2.5em; color: #4CAF50; text-align: center; }
#         .sidebar .sidebar-content { background-color: #f8f9fa; padding: 10px; border-radius: 10px; }
#         .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
#         .stTextInput>div>input { border-radius: 5px; }
#         .response-box { background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-top: 20px; }
#         .history-box { background-color: #f0f0f0; padding: 15px; border-radius: 10px; max-height: 400px; overflow-y: auto; }
#         .status-message { color: #4CAF50; font-style: italic; }
#         </style>
#     """, unsafe_allow_html=True)

#     # Initialize session state for history
#     if 'history' not in st.session_state:
#         st.session_state.history = []

#     st.markdown('<h1 class="main-title">📚 Chat with Your PDFs</h1>', unsafe_allow_html=True)
#     st.write("Ask questions about your PDFs by typing or speaking into your microphone! History is preserved below.")

#     col1, col2 = st.columns([2, 1])

#     with col1:
#         st.subheader("Ask a Question")

#         # Typed input
#         user_question_typed = st.text_input(
#             label="",
#             placeholder="Type your question here (e.g., 'What is the main topic of the PDFs?')",
#             key="text_input"
#         )
#         if st.button("Get Answer (Typed)", key="ask_button_typed"):
#             if user_question_typed:
#                 with st.spinner("Fetching answer..."):
#                     try:
#                         raw_answer = user_input(user_question_typed)
#                         formatted_answer = format_response(raw_answer)
#                         # Add to history
#                         st.session_state.history.append({
#                             "question": user_question_typed,
#                             "answer": formatted_answer,
#                             "source": "Typed"
#                         })
#                     except Exception as e:
#                         st.error(f"An error occurred: {str(e)}")
#             else:
#                 st.warning("Please type a question!")

#         # Audio input
#         st.write("Or use your microphone:")
#         audio = mic_recorder(
#             start_prompt="Start Recording",
#             stop_prompt="Stop Recording",
#             key="recorder",
#             format="wav"
#         )
        
#         if audio and 'bytes' in audio:
#             st.write("Transcribing your question...")
#             try:
#                 user_question = transcribe_audio(audio['bytes'])
#                 if not user_question.startswith("Sorry") and not user_question.startswith("Error"):
#                     with st.spinner("Fetching answer..."):
#                         raw_answer = user_input(user_question)
#                         formatted_answer = format_response(raw_answer)
#                         # Add to history
#                         st.session_state.history.append({
#                             "question": user_question,
#                             "answer": formatted_answer,
#                             "source": "Audio"
#                         })
#                 else:
#                     st.warning(user_question)
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")

#         # Display history
#         if st.session_state.history:
#             st.subheader("Question History")
#             history_text = ""
#             for i, entry in enumerate(st.session_state.history, 1):
#                 history_text += f"**Q{i} ({entry['source']}):** {entry['question']}\n\n"
#                 history_text += f"**A{i}:**\n{entry['answer']}\n\n---\n\n"
#                 st.markdown(f'<div class="history-box"><strong>Q{i} ({entry["source"]}):</strong> {entry["question"]}<br><strong>A{i}:</strong><br>{entry["answer"]}</div>', unsafe_allow_html=True)
            
#             # Download button
#             st.download_button(
#                 label="Download History",
#                 data=history_text,
#                 file_name="pdf_chatbot_history.txt",
#                 mime="text/plain",
#                 key="download_button"
#             )

#     with col2:
#         st.subheader("Upload PDFs")
#         pdf_docs = st.file_uploader(
#             "Drop your PDF files here",
#             type=["pdf"],
#             accept_multiple_files=True,
#             help="Upload one or more PDFs to process."
#         )
#         if st.button("Process PDFs", key="process_button"):
#             if pdf_docs:
#                 with st.spinner("Processing your PDFs..."):
#                     try:
#                         raw_text = get_pdf_text(pdf_docs)
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("PDFs processed successfully! You can now ask questions.")
#                     except Exception as e:
#                         st.error(f"Processing failed: {str(e)}")
#             else:
#                 st.warning("Please upload at least one PDF file!")
#         if os.path.exists("faiss_index"):
#             st.info("Vector store is ready. Ask away!")
#         else:
#             st.info("Upload and process PDFs to start chatting.")

#     st.markdown("---")
#     st.write("Built with ❤️ using Streamlit and Gemini by xAI")

# if __name__ == "__main__":
#     main()




# import os
# import google.generativeai as genai
# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain_core.prompts import PromptTemplate 
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from io import BytesIO
# import speech_recognition as sr
# from streamlit_mic_recorder import mic_recorder

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Existing functions (unchanged)
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_stream = BytesIO(pdf.getvalue())
#         pdf_reader = PdfReader(pdf_stream)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. Structure your response with key points listed as bullet points starting with "- ". Do not use bold formatting (**text**) in your response; keep it plain text. Provide all relevant details, and if the answer is not in the context, say, "answer is not available in the context". Do not provide incorrect information.\n\n
#     Context:\n{context}\n
#     Question:\n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]

# def format_response(response):
#     lines = response.split("\n")
#     formatted_response = ""
#     for line in lines:
#         if line.strip().startswith("- "):
#             parts = line.strip()[2:].split(":", 1)
#             if len(parts) == 2:
#                 keyword = parts[0].strip()
#                 description = parts[1].strip()
#                 formatted_response += f"- <b>{keyword}</b>: {description}\n"
#             else:
#                 formatted_response += f"{line}\n"
#         else:
#             formatted_response += f"{line}\n"
#     return formatted_response

# def transcribe_audio(audio_bytes):
#     recognizer = sr.Recognizer()
#     audio_file = BytesIO(audio_bytes)
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#         try:
#             text = recognizer.recognize_google(audio_data)
#             return text
#         except sr.UnknownValueError:
#             return "Sorry, I couldn't understand the audio."
#         except sr.RequestError as e:
#             return f"Error with speech recognition: {str(e)}"

# def main():
#     st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")
#     st.markdown("""
#         <style>
#         /* Main app container */
#         .stApp { 
#             background: linear-gradient(135deg, #0f172a, #1e293b); 
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
#             overflow: hidden; 
#             height: 100vh; 
#             padding: 20px; 
#             color: #e2e8f0; /* Light text for dark bg */
#         }
#         .main-title { 
#             font-size: 2em; 
#             color: #ffffff; 
#             text-align: center; 
#             background: linear-gradient(90deg, #06b6d4, #22d3ee); 
#             padding: 15px; 
#             border-radius: 10px; 
#             box-shadow: 0 2px 8px rgba(0,0,0,0.3); 
#             margin-bottom: 15px; 
#         }
#         .card { 
#             background: #1e293b; /* Dark slate card bg */
#             padding: 15px; 
#             border-radius: 10px; 
#             box-shadow: 0 2px 8px rgba(0,0,0,0.3); 
#             margin-bottom: 15px; 
#         }
#         .card:hover { 
#             transform: translateY(-3px); 
#         }
#         .stButton>button { 
#             background: linear-gradient(90deg, #06b6d4, #22d3ee); 
#             color: white; 
#             border: none; 
#             border-radius: 8px; 
#             padding: 8px 16px; 
#             font-weight: bold; 
#             font-size: 0.9em; 
#             transition: background 0.3s; 
#             margin-top: 10px; 
#         }
#         .stButton>button:hover { 
#             background: linear-gradient(90deg, #0e7490, #1aa3c1); 
#         }
#         .stTextInput>div>input { 
#             border-radius: 8px; 
#             border: 2px solid #06b6d4; 
#             padding: 8px; 
#             font-size: 0.9em; 
#             background: #1e293b; 
#             color: #e2e8f0; 
#             margin-bottom: 10px; 
#             width: 100%; /* Full width for typed input */
#         }
#         .history-box { 
#             background: #1e293b; 
#             padding: 12px; 
#             border-radius: 8px; 
#             max-height: 300px; 
#             overflow-y: auto; 
#             border-left: 4px solid #06b6d4; 
#             margin-bottom: 10px; 
#         }
#         .response-box { 
#             background: #334155; 
#             padding: 12px; 
#             border-radius: 8px; 
#             border-left: 4px solid #06b6d4; 
#         }
#         .status-message { 
#             color: #22d3ee; 
#             font-style: italic; 
#             font-size: 0.8em; 
#             margin-top: 5px; 
#         }
#         .icon { 
#             vertical-align: middle; 
#             margin-right: 8px; 
#         }
#         .footer { 
#             text-align: center; 
#             color: #94a3b8; /* Lighter gray for footer */
#             font-size: 0.8em; 
#             margin-top: 20px; 
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     # Initialize session state for history
#     if 'history' not in st.session_state:
#         st.session_state.history = []

#     # Main container
#     st.markdown('<div style="background: linear-gradient(135deg, #0f172a, #1e293b); height: 100vh; padding: 20px;">', unsafe_allow_html=True)
    
#     st.markdown('<h1 class="main-title">📚 PDF Chatbot</h1>', unsafe_allow_html=True)
#     st.write('<div style="text-align: center; color: #cbd5e1; font-size: 1em; margin-bottom: 15px;">Ask your PDFs anything!</div>', unsafe_allow_html=True)

#     col1, col2 = st.columns([2, 1], gap="medium")

#     with col1:
#         st.markdown('<div class="card"><h2 style="color: #06b6d4; font-size: 1.5em;">Ask a Question</h2>', unsafe_allow_html=True)

#         # Typed input
#         st.write('<span class="icon">✍️</span>Type your question:', unsafe_allow_html=True)
#         user_question_typed = st.text_input(
#             label="",
#             placeholder="e.g., What is the main topic?",
#             key="text_input"
#         )
#         if st.button("🔍 Get Answer", key="ask_button_typed"):
#             if user_question_typed:
#                 with st.spinner("Fetching answer..."):
#                     try:
#                         raw_answer = user_input(user_question_typed)
#                         formatted_answer = format_response(raw_answer)
#                         st.session_state.history.append({
#                             "question": user_question_typed,
#                             "answer": formatted_answer,
#                             "source": "Typed"
#                         })
#                     except Exception as e:
#                         st.error(f"An error occurred: {str(e)}")
#             else:
#                 st.warning("Please type a question!")

#         # Audio input (no text input box)
#         st.write('<span class="icon">🎤</span>Speak your question:', unsafe_allow_html=True)
#         audio = mic_recorder(
#             start_prompt="🎙️ Start",
#             stop_prompt="⏹️ Stop",
#             key="recorder",
#             format="wav"
#         )
        
#         if audio and 'bytes' in audio:
#             st.write('<p class="status-message">Transcribing...</p>', unsafe_allow_html=True)
#             try:
#                 user_question = transcribe_audio(audio['bytes'])
#                 if not user_question.startswith("Sorry") and not user_question.startswith("Error"):
#                     with st.spinner("Fetching answer..."):
#                         raw_answer = user_input(user_question)
#                         formatted_answer = format_response(raw_answer)
#                         st.session_state.history.append({
#                             "question": user_question,
#                             "answer": formatted_answer,
#                             "source": "Audio"
#                         })
#                 else:
#                     st.warning(user_question)
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
        
#         st.markdown('</div>', unsafe_allow_html=True)

#         # History section
#         if st.session_state.history:
#             st.markdown('<div class="card"><h2 style="color: #06b6d4; font-size: 1.5em;">History</h2>', unsafe_allow_html=True)
#             history_text = ""
#             for i, entry in enumerate(st.session_state.history, 1):
#                 history_text += f"**Q{i} ({entry['source']}):** {entry['question']}\n\n"
#                 history_text += f"**A{i}:**\n{entry['answer']}\n\n---\n\n"
#                 st.markdown(f'<div class="history-box"><strong>Q{i} ({entry["source"]}):</strong> {entry["question"]}<br><strong>A{i}:</strong><br>{entry["answer"]}</div>', unsafe_allow_html=True)
            
#             st.download_button(
#                 label="📥 Download",
#                 data=history_text,
#                 file_name="pdf_chatbot_history.txt",
#                 mime="text/plain",
#                 key="download_button"
#             )
#             st.markdown('</div>', unsafe_allow_html=True)

#     with col2:
#         st.markdown('<div class="card"><h2 style="color: #06b6d4; font-size: 1.5em;">Upload PDFs</h2>', unsafe_allow_html=True)
#         pdf_docs = st.file_uploader(
#             "Drop PDFs here",
#             type=["pdf"],
#             accept_multiple_files=True,
#             help="Upload PDFs to process."
#         )
#         if st.button("📤 Process", key="process_button"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     try:
#                         raw_text = get_pdf_text(pdf_docs)
#                         text_chunks = get_text_chunks(raw_text)
#                         get_vector_store(text_chunks)
#                         st.success("PDFs processed!")
#                     except Exception as e:
#                         st.error(f"An error occurred: {str(e)}")
#             else:
#                 st.warning("Upload at least one PDF!")
#         if os.path.exists("faiss_index"):
#             st.info("✅ Ready to ask!")
#         else:
#             st.info("📋 Process PDFs to start.")
#         st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown('<div class="footer">Built with ❤️ by xAI</div>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)  # Close background div

# if __name__ == "__main__":
#     main()









import os
import google.generativeai as genai
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Existing functions (unchanged)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_stream = BytesIO(pdf.getvalue())
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Structure your response with key points listed as bullet points starting with "- ". Do not use bold formatting (**text**) in your response; keep it plain text. Provide all relevant details, and if the answer is not in the context, say, "answer is not available in the context". Do not provide incorrect information.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def format_response(response):
    lines = response.split("\n")
    formatted_response = ""
    for line in lines:
        if line.strip().startswith("- "):
            parts = line.strip()[2:].split(":", 1)
            if len(parts) == 2:
                keyword = parts[0].strip()
                description = parts[1].strip()
                formatted_response += f"- <b>{keyword}</b>: {description}\n"
            else:
                formatted_response += f"{line}\n"
        else:
            formatted_response += f"{line}\n"
    return formatted_response

def transcribe_audio(audio_bytes):
    recognizer = sr.Recognizer()
    audio_file = BytesIO(audio_bytes)
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError as e:
            return f"Error with speech recognition: {str(e)}"

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")
    st.markdown("""
        <style>
        /* Main app container */
        .stApp { 
            background: linear-gradient(135deg, #0f172a, #1e293b); 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            overflow: hidden; 
            height: 100vh; 
            padding: 20px; 
            color: #e2e8f0; /* Light text for dark bg */
        }
        .main-title { 
            font-size: 2em; 
            color: #ffffff; 
            text-align: center; 
            background: linear-gradient(90deg, #06b6d4, #22d3ee); 
            padding: 15px; 
            border-radius: 10px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.3); 
            margin-bottom: 15px; 
        }
        .card { 
            background: #1e293b; /* Dark slate card bg */
            padding: 15px; 
            border-radius: 10px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.3); 
            margin-bottom: 15px; 
        }
        .card:hover { 
            transform: translateY(-3px); 
        }
        .stButton>button { 
            background: linear-gradient(90deg, #06b6d4, #22d3ee); 
            color: white; 
            border: none; 
            border-radius: 8px; 
            padding: 8px 16px; 
            font-weight: bold; 
            font-size: 0.9em; 
            transition: background 0.3s; 
            margin-top: 10px; 
        }
        .stButton>button:hover { 
            background: linear-gradient(90deg, #0e7490, #1aa3c1); 
        }
        /* Updated CSS for larger text input box */
        .stTextInput>div>input { 
            border-radius: 8px; 
            border: 2px solid #06b6d4; 
            padding: 12px; /* Increased padding */
            font-size: 1.2em; /* Larger font size */
            background: #1e293b; 
            color: #e2e8f0; 
            margin-bottom: 10px; 
            width: 100%; /* Full width for typed input */
            height: 80px; /* Increased height */
            resize: none; /* Prevent manual resizing */
        }
        .history-box { 
            background: #1e293b; 
            padding: 12px; 
            border-radius: 8px; 
            max-height: 300px; 
            overflow-y: auto; 
            border-left: 4px solid #06b6d4; 
            margin-bottom: 10px; 
        }
        .response-box { 
            background: #334155; 
            padding: 12px; 
            border-radius: 8px; 
            border-left: 4px solid #06b6d4; 
        }
        .status-message { 
            color: #22d3ee; 
            font-style: italic; 
            font-size: 0.8em; 
            margin-top: 5px; 
        }
        .icon { 
            vertical-align: middle; 
            margin-right: 8px; 
        }
        .footer { 
            text-align: center; 
            color: #94a3b8; /* Lighter gray for footer */
            font-size: 0.8em; 
            margin-top: 20px; 
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Main container
    st.markdown('<div style="background: linear-gradient(135deg, #0f172a, #1e293b); height: 100vh; padding: 20px;">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">📚 PDF Chatbot</h1>', unsafe_allow_html=True)
    st.write('<div style="text-align: center; color: #cbd5e1; font-size: 1em; margin-bottom: 15px;">Ask your PDFs anything!</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="medium")

    with col1:
        st.markdown('<div class="card"><h2 style="color: #06b6d4; font-size: 1.5em;">Ask a Question</h2>', unsafe_allow_html=True)

        # Typed input (larger)
        st.write('<span class="icon">✍️</span>Type your question:', unsafe_allow_html=True)
        user_question_typed = st.text_input(
            label="",
            placeholder="e.g., What is the main topic?",
            key="text_input"
        )
        if st.button("🔍 Get Answer", key="ask_button_typed"):
            if user_question_typed:
                with st.spinner("Fetching answer..."):
                    try:
                        raw_answer = user_input(user_question_typed)
                        formatted_answer = format_response(raw_answer)
                        st.session_state.history.append({
                            "question": user_question_typed,
                            "answer": formatted_answer,
                            "source": "Typed"
                        })
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please type a question!")

        # Audio input (no text input box)
        st.write('<span class="icon">🎤</span>Speak your question:', unsafe_allow_html=True)
        audio = mic_recorder(
            start_prompt="🎙️ Start",
            stop_prompt="⏹️ Stop",
            key="recorder",
            format="wav"
        )
        
        if audio and 'bytes' in audio:
            st.write('<p class="status-message">Transcribing...</p>', unsafe_allow_html=True)
            try:
                user_question = transcribe_audio(audio['bytes'])
                if not user_question.startswith("Sorry") and not user_question.startswith("Error"):
                    with st.spinner("Fetching answer..."):
                        raw_answer = user_input(user_question)
                        formatted_answer = format_response(raw_answer)
                        st.session_state.history.append({
                            "question": user_question,
                            "answer": formatted_answer,
                            "source": "Audio"
                        })
                else:
                    st.warning(user_question)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # History section
        if st.session_state.history:
            st.markdown('<div class="card"><h2 style="color: #06b6d4; font-size: 1.5em;">History</h2>', unsafe_allow_html=True)
            history_text = ""
            for i, entry in enumerate(st.session_state.history, 1):
                history_text += f"**Q{i} ({entry['source']}):** {entry['question']}\n\n"
                history_text += f"**A{i}:**\n{entry['answer']}\n\n---\n\n"
                st.markdown(f'<div class="history-box"><strong>Q{i} ({entry["source"]}):</strong> {entry["question"]}<br><strong>A{i}:</strong><br>{entry["answer"]}</div>', unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Download",
                data=history_text,
                file_name="pdf_chatbot_history.txt",
                mime="text/plain",
                key="download_button"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h2 style="color: #06b6d4; font-size: 1.5em;">Upload PDFs</h2>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader(
            "Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDFs to process."
        )
        if st.button("📤 Process", key="process_button"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Upload at least one PDF!")
        if os.path.exists("faiss_index"):
            st.info("✅ Ready to ask!")
        else:
            st.info("📋 Process PDFs to start.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Built with ❤️ by xAI</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close background div

if __name__ == "__main__":
    main()