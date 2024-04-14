'''
TO-DOs:
- This model uses Hugging Face API
- Currently working!
'''

import streamlit as st
from operator import itemgetter
import os
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, ConversationChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory

from langchain.chains import (
    StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
)
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI


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


history_chat = []


def main():
    # Chat configuration
    st.header(f"Welcome to {company_name} Assist Tool")
    print("Esta imprimiendo esto")

    load_dotenv()

    # Load Embeddigs DB
    index_name = "thesis-model-1"  # Existing Pinecone index
    emb_model = "sentence-transformers/all-mpnet-base-v2"  # Encoder
    embeddings = HuggingFaceHubEmbeddings(repo_id=emb_model,
                                          huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))

    # Load Embeddings from existing Pinecone index
    vectorStore = Pinecone.from_existing_index(index_name, embeddings)
    retriever = vectorStore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3})

    # Load Azure OpenAI
    llm_model = load_model()

    # Prompt

    template = """ You are a chatbot ready to help person with their queries based on a context and chat history.

  
        QUESTION: 
        {input}

        CHAT HISTORY: 
        {history}
  
        ANSWER:
        """

    prompt = PromptTemplate(
        input_variables=["history", "query", "context"], template=template)
    # memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

    # Initialize Model Messages

    # Conversation memory using summary
    memory = ConversationSummaryMemory.from_messages(llm=llm_model,
                                                     chat_memory=history_chat)
    conversation_with_summary = ConversationChain(
        llm=llm_model,
        memory=memory,
        verbose=True,
    )

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
        history_chat.add_user_message(query)
        docs = vectorStore.similarity_search(query=query, k=3)
        # st.write(f'memory {memory.buffer}')
        response = conversation_with_summary.predict(input=query)
        print(f'estos son los mensajes {history_chat.messages}')
        print(f'streamlit messages {
              [x.values() for x in st.session_state.messages]}')
        print(f'this is the memory {memory.buffer}')
        # User query
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Model response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
