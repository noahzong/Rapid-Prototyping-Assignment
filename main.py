import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import *
from langchain.chains.summarize import load_summarize_chain
import tempfile
from langchain.docstore.document import Document

def assignmentBreakdown(assignment_description, final_deadline, start_date):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    system_template = """You are an assistant tasked with breaking down an assignment into smaller chunks with shorter deadlines."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """The assignment is described as follows: {assignment_description}. The final deadline is {final_deadline}. Please create a breakdown of the assignment into smaller chunks with shorter deadlines, starting from {start_date}."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(assignment_description=assignment_description, final_deadline=final_deadline, start_date=start_date)
    return result # returns string   

st.title("Deadlines")

# Initialize user inputs
assignment_description = ""
final_deadline = ""
start_date = ""

# Get the assignment description from the user
assignment_description = st.text_area("Assignment Description")

# Get the final deadline from the user
final_deadline = st.text_input("Final Deadline")

# Get the start date from the user
start_date = st.text_input("Start Date")

# Display the breakdown of the assignment to the user
if st.button("Submit"):
    if assignment_description and final_deadline and start_date:
        breakdown = assignmentBreakdown(assignment_description, final_deadline, start_date)
        st.markdown(breakdown)
    else:
        st.markdown("")
