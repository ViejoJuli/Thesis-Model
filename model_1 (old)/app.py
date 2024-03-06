'''
TO-DOs:
- Creaate a chat model from langchain.chat_models import ChatOpenAI
- Keep context in chat

'''

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pickle
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Sidebar contents
with st.sidebar:
    st.title('Chatbot')
    st.markdown('''
        ## About
        This app is a Chatbot''')
    add_vertical_space(5)
    st.write('Made with Love')

# # account for deprecation of LLM model
# import datetime
# # Get the current date
# current_date = datetime.datetime.now().date()

# # Define the date after which the model should be set to "gpt-3.5-turbo"
# target_date = datetime.date(2024, 6, 12)

# # Set the model variable based on the current date
# if current_date > target_date:
#     llm_model = "gpt-3.5-turbo"
# else:
#     llm_model = "gpt-3.5-turbo-0301"


def main():
    st.header("Welcome to Assist Tool")

    load_dotenv()

    # If I need a PDF file uploader
    # pdf = st.file_upload("Upload your PDF", type="pdf")

    # Reads PDF file
    # if pdf is not None:
    pdf_reader = PdfReader("user manual.pdf")

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text in smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,  # Keep context
        length_function=len,
    )
    chunks = text_splitter.split_text(text=text)

    # st.write(chunks)

    # Computing embeddings
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    # Store the embeddings
    if os.path.exists('document.pkl'):
        with open('document.pkl', 'rb') as f:
            VectorStore = pickle.load(f)
        st.write('Embeddings loaded from memory')
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open('document.pkl', 'wb') as f:
            pickle.dump(VectorStore, f)
        st.write('Embeddings run and completed successfully')

    # Accept user comment
    query = st.text_input("Ingrese su inquietud")
    st.write(query)

    # Find info in Knowledge database
    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        st.write(docs)  # write nearest chunks
        llm = OpenAI(temperature=0.9, model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            # TODO: Modify prompts, and review current prompts

            # To know costs
            print(cb)
        st.write(response)


if __name__ == '__main__':
    main()
