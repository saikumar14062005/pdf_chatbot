
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import os
from text_processing import get_pdf_text, get_text_chunks
from vector_storage import get_vector_store
from chatbot import user_input, format_response
from audio_processing import transcribe_audio

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")
    st.markdown("""
        <style>
        /* Remove default Streamlit padding and margins */
        .stApp {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #e2e8f0;
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden; /* Disable scrolling */
            display: flex;
            flex-direction: column;
        }
        /* Ensure body and html have no scroll */
        html, body {
            margin: 0;
            padding: 0;
            height: 100%;
            overflow: hidden;
        }
        /* Main container styling */
        .main-container {
            flex: 1; /* Take available space */
            padding: 20px;
            margin: 0;
            display: flex;
            flex-direction: column;
            justify-content: flex-start; /* Align content to top */
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
            background: #1e293b;
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
        .stDownloadButton>button {
            background: linear-gradient(90deg, #10b981, #34d399);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 0.9em;
            transition: background 0.3s;
            margin-top: 10px;
        }
        .stDownloadButton>button:hover {
            background: linear-gradient(90deg, #047857, #10b981);
        }
        .stTextInput>div>input { 
            border-radius: 8px; 
            border: 2px solid #06b6d4; 
            padding: 12px;
            font-size: 1.2em;
            background: #1e293b; 
            color: #e2e8f0; 
            margin-bottom: 10px; 
            width: 100%;
            height: 80px;
            resize: none;
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
            color: #94a3b8;
            font-size: 0.8em; 
            margin-top: 20px; 
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Main container with corrected class
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">📚 PDF Chatbot</h1>', unsafe_allow_html=True)
    st.write('<div style="text-align: center; color: #cbd5e1; font-size: 1em; margin-bottom: 15px;">Ask your PDFs anything!</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="medium")

    with col1:
        st.markdown('<div class="card"><h2 style="color: #06b6d4; font-size: 1.5em;">Ask a Question</h2>', unsafe_allow_html=True)

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
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

if __name__ == "__main__":
    import config
    main()