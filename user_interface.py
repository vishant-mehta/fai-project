import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

openai_api_key = 'Your OpenAI API Key'# Initialize chatbot with the OpenAI API key
chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2, openai_api_key=openai_api_key)

# Define the system message prompt
system_message_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI assistant that helps determine if an answer is relevant to a question in a given context." 
    "Provide feedback on the answer's relevance and score it out of 10 . Only give the score and feedback in bold in seperate paragraphs with a line space in between and don't give anything extra. "
    "Write score out of 10 first and then Feedback\n\nScore:\n\nFeedback:"
)

# Define the human message prompt
human_message_prompt = HumanMessagePromptTemplate.from_template(
    "Question: {question}\nAnswer: {answer}\nContext: {context}\n\nIs the answer relevant to the question in the given context? Please provide feedback on the answer's relevance, mention any important points missed, and score the answer out of 5. Explain your reasoning."
)

# Create the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

def main():
    st.set_page_config(page_title="IntelliGrade", page_icon=":mortar_board:")
    
    # Custom CSS styles
    st.markdown(
        """
        <style>

        body {
            background-color: while;
        }
        .stTextInput label, .stTextArea label {
            color: #1E90FF;
            font-weight: bold;
        }
        .stButton button {
            background-color: #1E90FF;
            color: white;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title(":rainbow[IntelliGrade: Enhancing Descriptive Answer Assessment with LLMs :mortar_board:]")
    
    # Get user input
    context = st.text_area("Enter the context:")
    question = st.text_input("Enter the question:")
    answer = st.text_input("Enter the answer:")
    
    
    if st.button("Check Relevance :mag:"):
        # Generate the chat response
        chat_response = chat(chat_prompt.format_prompt(question=question, answer=answer, context=context).to_messages())
        
        # Display the chat response
        st.write("Assistant: " + chat_response.content)

if __name__ == "__main__":
    main()