# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain_core.prompts import PromptTemplate
# from vector_storage import load_vector_store

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. Structure your response with key points listed as bullet points starting with "- ". If the answer is not in the context, say, "answer is not available in the context".
    
#     Context:
#     {context}
    
#     Question:
#     {question}
    
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def user_input(user_question):
#     vector_store = load_vector_store()
#     docs = vector_store.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]



from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

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