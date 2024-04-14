'''
1st model: OpenAI and
'''

import streamlit as st
from operator import itemgetter
import os
from dotenv import load_dotenv


from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import Pinecone

company_name = "Leal"


# Sidebar contents
with st.sidebar:
    st.title(f'{company_name} Chatbot')

    # Concise and clear description
    st.markdown(f"Este es un chatbot diseñado para responder las preguntar al respecto de {
                company_name}")

    st.write("---")

    st.write("Hecho con ❤️ por Julián David Ríos López")


def load_model():
    llm_model = 'gpt-35-turbo-jdrios'
    llm = AzureChatOpenAI(model_name=llm_model,
                          temperature=0,
                          api_version="2023-09-15-preview",
                          azure_endpoint=os.getenv('OPENAI_ENDPOINT'))
    return llm


def main():
    # Chat configuration
    st.header(f"Bienvenido al asistente virtual de {company_name}")

    load_dotenv()

    index_name = "thesis-model-1"  # Existing Pinecone index
    emb_model = 'text-embedding-ada-002-jdrios'
    # # Load Embeddigs DB (Hugging Face)

    # emb_model = "sentence-transformers/all-mpnet-base-v2"  # Encoder
    # embeddings = HuggingFaceHubEmbeddings(repo_id=emb_model,
    #                                       huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))

    # Load Embeddings DB (OpenAI)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=emb_model, azure_endpoint=os.getenv('OPENAI_ENDPOINT'))

    # Load Embeddings from existing Pinecone index
    vector_store = Pinecone.from_existing_index(index_name, embeddings)

    # Load Azure OpenAI
    llm_model = load_model()

    # Prompt

    template = """ You are a chatbot ready to help person with their queries based on a context and chat history.
    If the person gives you his name, please use it in all the responses. 
        CONTEXT:
        {context}
  
        QUESTION: 
        {query}

        CHAT HISTORY: 
        {chat_history}
  
        ANSWER:
        """

    prompt = PromptTemplate(
        input_variables=["chat_history", "query", "context"], template=template)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": f"Bienvenido al chat de {
            company_name}, soy su asistente virtual, en que le puedo ayudar el día de hoy?"})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # put the role in the icon
            st.markdown(message["content"])

    # Initialize Model Messages

    # Conversation memory using summary
    # memory = ConversationSummaryMemory.from_messages(llm=llm_model,
    #                                                  chat_memory=history_1)
    chain = load_qa_chain(llm=llm_model,
                          chain_type="stuff",
                          prompt=prompt,
                          verbose=True)

    # Initialize chat

    # React to user input
    if query := st.chat_input("Cual es tu consulta"):
        # chat_history = " ".join([message["content"]
        #                         for message in st.session_state.messages[-3:]])
        # print(f"before {chat_history}")
        docs = vector_store.similarity_search(query=query, k=3)
        # st.write(f'memory {memory.buffer}')
        response = chain({"query": query,
                          "input_documents": docs,
                          "chat_history": st.session_state.messages
                          })['output_text']
        # print(f'this is the memory {memory.buffer}')
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
