'''
TO-DOs:
- Creaate a chat model from langchain.chat_models import ChatOpenAI
- Keep context in chat

'''

import streamlit as st
import openai
from dotenv import load_dotenv
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain_pinecone import Pinecone

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
    openai.api_type = "azure"
    openai.api_base = os.getenv('OPENAI_ENDPOINT')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.api_version = "2023-09-15-preview"
    llm_model = 'gpt-35-turbo-jdrios'
    llm = OpenAI(temperature=0,  engine=llm_model)
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

    # Load Azure OpenAI
    llm_model = load_model()

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

        # User query
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Model response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = "Aqui va la respuesta"
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})

    # Accept user comment

    query = ""
    # Find info in Knowledge database
    if query:
        docs = vectorStore.similarity_search(query=query, k=3)
        st.write(docs)  # write nearest chunks
        chain = load_qa_chain(llm=llm_model, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            # TODO: Modify prompts, and review current prompts

            # To know costs
            print(cb)
        st.write(response)


if __name__ == '__main__':
    main()
