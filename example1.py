## Integrate our code with OpenAI API
import os
from constraints import openai_keys
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain #for executing prompt
from langchain.chains import SequentialChain
import streamlit as st


#Initialize openAI key
os.environ["OPENAI_API_KEY"] = openai_keys

# Streamlit framework
st.title('Langchain Demo with OpenAI')
st.title('Data Science concepts search')
input_text=st.text_input('Search the topic you want')

# Prompt Template
first_input_prompt = PromptTemplate(
    input_variables=['Topic'],
    template="You are expert data science professor who is proficient in explaining things in a intuitive way with simple example. Now  explain about this concept {Topic}"
)



##OPEN AI LLMs -2 
llm = OpenAI(temperature=0.8) # between 0 to 1, (control agent should have while giving response/balanced answer)

# Output of one template should go to next template
chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='Intuition')

second_input_prompt = PromptTemplate(
    input_variables=['Intuition'],
    template="You are expert data science professor who is proficient in explaining things in a intuitive way with simple examples and mathematic explanation. Now  explain mathematics behind {Intuition} using a small dummy dataset"
)

chain2 = LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='Mathematical_View')


third_input_prompt = PromptTemplate(
    input_variables=['Topic'],
    template="You are expert data science professor who is proficient in explaining things in a intuitive way with simple examples and mathematic explanation. Now  state three critical applications where this {Mathematical_View} is used"
)

chain3 = LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='Applications')

parent_chain = SequentialChain(
    chains=[chain,chain2,chain3],input_variables=['Topic'],output_variables=['Intuition','Mathematical_View','Applications'],verbose=True)

if input_text: #if input_text is yes.
    st.write(parent_chain({'Topic':input_text}))


