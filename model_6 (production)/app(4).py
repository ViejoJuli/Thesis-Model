'''
TO-DOs:
- This model uses Hugging Face API}
- This model uses ConversationRetrievalChain

'''

import streamlit as st
from operator import itemgetter
import os
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.question_answering import load_qa_chain

company_name = "Leal"


# Sidebar contents
with st.sidebar:
    st.title('Chatbot')
    st.markdown('''
        ## About
        This app is a Chatbot''')
    add_vertical_space(5)
    st.write('Made with Love')


def load_model():
    llm_model = 'gpt-35-turbo-jdrios'
    llm = AzureChatOpenAI(model_name=llm_model,
                          temperature=0,
                          api_version="2023-09-15-preview",
                          azure_endpoint=os.getenv('OPENAI_ENDPOINT'))
    return llm


def main():
    # Chat configuration
    st.header(f"Welcome to {company_name} Assist Tool")

    load_dotenv()

    # Load Embeddigs DB
    index_name = "thesis-model-1"  # Existing Pinecone index
    emb_model = "sentence-transformers/all-mpnet-base-v2"  # Encoder
    embeddings = HuggingFaceHubEmbeddings(repo_id=emb_model,
                                          huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))
    # # embeddings from OpenAI
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

    # Load Embeddings from existing Pinecone index
    vectorStore = Pinecone.from_existing_index(index_name, embeddings)
    retriever = vectorStore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3})

    # Load Azure OpenAI
    llm_model = load_model()
    chat_history = []

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": f"Bienvenido al chat de {
            company_name}, soy su asistente virtual, en que le puedo ayudar el día de hoy?"})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # put the role in the icon
            st.markdown(message["content"])

    # Initialize chat

    # React to user input
    if query := st.chat_input("Cual es tu consulta"):

        chain = load_qa_chain(llm=llm_model, chain_type="stuff")
        qa = ConversationalRetrievalChain.from_llm(
            llm_model, retriever, return_source_documents=True)
        result = qa({"question": query, "chat_history": chat_history})
        response = result["answer"]
        chat_history.append((query, response))

        # write nearest chunks
        st.write(result['source_documents'][0].page_content)

        # Write user query in UI
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Write model response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
