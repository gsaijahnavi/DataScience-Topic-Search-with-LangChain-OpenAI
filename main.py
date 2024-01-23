import os
from constraints import openai_keys
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Streamlit framework
st.title('Data Science topic search with LangChain OpenAI')
st.markdown("<style>h1 { font-size: 24px !important; }</style>", unsafe_allow_html=True)
background_image = "https://i.pinimg.com/564x/6e/cc/21/6ecc21e07579ac1fb3224fdfbf25851b.jpg"
st.markdown(
    f"""
    <style>
        body {{
            background-image: url('{background_image}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
        }}
        .stApp {{
            background: none;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Input box for OpenAI key
openai_key = st.text_input('Enter your OpenAI key:')
if not openai_key:
    st.warning('Please enter your OpenAI key to proceed.')

else:
    input_text = st.text_input('Search any data science topic you want')

    # Prompt Template
    first_input_prompt = PromptTemplate(
        input_variables=['Topic'],
        template="You are an expert data science professor who is proficient in explaining things in an intuitive way with simple examples. Now explain about this concept {Topic}"
    )

    # OPEN AI LLMs -2
    llm = OpenAI(temperature=0.8, api_key=openai_key)  # Pass the OpenAI key to authenticate

    # Output of one template should go to the next template
    chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='Intuition')

    second_input_prompt = PromptTemplate(
        input_variables=['Intuition'],
        template="You are an expert data science professor who is proficient in explaining things in an intuitive way with simple examples and mathematical explanation. Now explain the mathematics behind {Intuition} using a small dummy dataset"
    )

    chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='Mathematical_View')

    third_input_prompt = PromptTemplate(
        input_variables=['Topic'],
        template="You are an expert data science professor who is proficient in explaining things in an intuitive way with simple examples and mathematical explanation. Now state three critical applications where this {Mathematical_View} is used"
    )

    chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='Applications')

    parent_chain = SequentialChain(
        chains=[chain, chain2, chain3], input_variables=['Topic'], output_variables=['Intuition', 'Mathematical_View', 'Applications'], verbose=True)

    if input_text:  # if input_text is yes.
        st.write(parent_chain({'Topic': input_text}))
