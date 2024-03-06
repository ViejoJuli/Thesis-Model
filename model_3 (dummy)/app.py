from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import os
import requests
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def img2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base")  # task, model
    text = image_to_text(url)[0]['generated_text']

    print(text)
    return text


def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a story from a simple narrative, the story should not be more than 20 words;
    CONTEXT: {scenario}
    STORY: 
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    llm_model = HuggingFaceHub(repo_id='google/gemma-2b-it',
                               model_kwargs={"temperature": 0.1, "max_new_tokens": 200})
    story_llm = LLMChain(llm=llm_model,
                         prompt=prompt,
                         verbose=True)
    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    print("Response from API received")
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="😎😎😎")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file:  # is not None
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.',
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")


if __name__ == '__main__':
    main()
