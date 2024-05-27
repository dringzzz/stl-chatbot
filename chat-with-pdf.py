from PyPDF2 import PdfReader
from io import StringIO

import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import streamlit as st


genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
dl_dir = 'transcripts/'


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

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        txt_file = st.file_uploader(
            "Upload your file(s) and Click on the Submit & Process Button", accept_multiple_files=False)
        if st.button("ประมวลผล!"):
            with st.spinner("Processing..."):
                # raw_text = get_pdf_text(pdf_docs) # Replace with getting raw string from text file
                stringio = StringIO(txt_file.getvalue().decode("utf-8"))
                raw_text = stringio.read()

                # raw_text = ""
                # for file in glob.glob(dl_dir + "/*.txt"):
                #     print(file)
                #
                #     my_file = open(file)
                #     raw_text += my_file.read()
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

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
