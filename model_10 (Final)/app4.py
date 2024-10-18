'''
1st model: OpenAI and
'''

import requests
import streamlit as st  # for interface
import os
from dotenv import load_dotenv
import pyodbc
from pysentimiento import create_analyzer  # for sentiment analysis
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import Pinecone
from langchain_community.callbacks import get_openai_callback  # for costs

load_dotenv()

# For summarization


def summary_query(payload):
    API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert2bert_shared-spanish-finetuned-summarization"
    headers = {"Authorization": f"Bearer {
        os.environ['HUGGINGFACEHUB_API_TOKEN']}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


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


def get_sentiment(text):
    analyzer = create_analyzer(task="sentiment", lang="es")
    sentiment = analyzer.predict(text).output
    if sentiment == "POS":
        sentiment = "positive"
    elif sentiment == "NEU":
        sentiment = "neutral"
    else:
        sentiment = "negative"
    return sentiment


def main():
    # Chat configuration
    st.header(f"Bienvenido al asistente virtual de {company_name}")

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

    template = """ You are a customer service chatbot ready to help person with their queries based on a context and chat history.
    If the person gives you his name, please use it in all the responses. Take in count the
    person's sentiment to adjust your tone.
        CONTEXT:
        {context}
  
        QUESTION: 
        {query}

        CHAT HISTORY: 
        {chat_history}

        SENTIMENT:
        {sentiment}
  
        ANSWER:
        """

    prompt = PromptTemplate(
        input_variables=["chat_history", "query", "context", "sentiment"], template=template)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": f"Bienvenido al chat de {
            company_name}, soy su asistente virtual, en que le puedo ayudar el día de hoy?"})

    # Initialize costs
    if 'costs' not in st.session_state:
        st.session_state['costs'] = 0

    # Initialize sentiment
    if 'sentiment' not in st.session_state:
        st.session_state['sentiment'] = 'neutral'

    # # Conection to DB
    # conn = st.connection("postgresql", type="sql")

    # # Perform query.
    # df = conn.query('SELECT * FROM dummy;', ttl="10m")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # put the role in the icon
            st.markdown(message["content"])

    # Initialize Chain
    chain = load_qa_chain(llm=llm_model,
                          chain_type="stuff",
                          prompt=prompt,
                          verbose=True)

    # Initialize chat
    # React to user input
    if query := st.chat_input("Cual es tu consulta"):

        # Append user message to session_state
        st.session_state.messages.append({"role": "user", "content": query})

        # Total interactions
        history_string = " ".join([message["content"]
                                   for message in st.session_state.messages[-3:]])

        # User interactions
        user_history_string = " ".join([message["content"]
                                        for message in st.session_state.messages[-3:] if message["role"] == "user"])
        docs = vector_store.similarity_search(query=query, k=3)
        st.session_state['sentiment'] = get_sentiment(user_history_string)
        print(st.session_state['sentiment'])

        with get_openai_callback() as cb:
            response = chain({"query": query,
                              "input_documents": docs,
                              "chat_history": history_string,
                              "sentiment": st.session_state['sentiment']
                              })['output_text']

        # print("Successful requests:", callback_handler.successful_requests)
        # print("Total tokens:", callback_handler.total_tokens)
        # print("Prompt tokens:", callback_handler.prompt_tokens)
        # print("Completion tokens:", callback_handler.completion_tokens)
        # print("Total cost:", callback_handler.total_cost)

        # Update costs
        st.session_state['costs'] += cb.total_cost
        print(f"costs are {st.session_state['costs']}")

        with st.chat_message("user"):
            st.markdown(query)

        # Model response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
