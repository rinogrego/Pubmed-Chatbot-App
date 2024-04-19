import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(
    page_title="Chatbot App",
    layout="centered"
)

st.title("Chatbot app")
st.markdown("---")


# get LLM response
def get_response(query, chat_history, stream=False):
    template = """
    You are a helpful assistant. Answer the following questions considering the chat history..
    You like to give answer by first write a comprehensive explanation and then use bullet points for the most important parts.
    
    chat history: {chat_history}
    
    user question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # using google gemini, the following error occured: https://github.com/google/generative-ai-python/issues/196
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
    # llm = ChatOpenAI()
    
    chain = prompt | llm | StrOutputParser()
    
    params = dict(chat_history=chat_history, user_question=query)
    if stream:
        return chain.stream(params)
    else:
        return chain.invoke(params)

def main(): 
    # display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

    # user input
    user_query = st.chat_input("What's on your mind?")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            # ai_response = "Halo"
            # st.markdown(ai_response)
            ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history, stream=True))
            
        st.session_state.chat_history.append(AIMessage(ai_response))
    
if __name__ == "__main__":
    main()