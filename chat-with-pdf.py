import google.generativeai as genai
import glob
import os

from PyPDF2 import PdfReader
from io import StringIO

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import streamlit as st


genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
dl_dir = 'transcripts/'


def read_file_transcripts_folder(category: str):
    st.success(category)
    raw_text = ""
    for file in glob.glob(dl_dir + "/*.txt"):
        my_file = open(file)
        raw_text += my_file.read()
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """ช่วยตอบคำถามให้ละเอียดที่สุดเท่าที่จะทำได้"\n\n Context:\n {context}?\n Question: \n{question}\n 

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "เริ่มแชทกับ Conicle AI ได้เลย!"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="ConicleAI Chatbot",
        page_icon="🤖"
    )

    read_file_transcripts_folder('Default')

    # Sidebar
    with st.sidebar:
        # Define the menu options
        menu_options = ["Home", "Upload PDF", "Process Data", "About"]
        # Create a selectbox for the menu
        menu_selection = st.selectbox("Category", menu_options)
        # Display different content based on the menu selection
        if menu_selection == "Home":
            read_file_transcripts_folder('Home')
        elif menu_selection == "Upload PDF":
            read_file_transcripts_folder('Upload PDF')
        elif menu_selection == "Process Data":
            read_file_transcripts_folder('Process Data')
        elif menu_selection == "About":
            read_file_transcripts_folder('About')

    # Main content area for displaying chat messages
    st.title("ConicleAI Chatbot")
    st.write("อยากคุย!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "เริ่มแชทกับ Conicle AI ได้เลย!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(user_question=prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
