import os
from dotenv import load_dotenv
import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Function to initialize environment variables and configure settings
def initialize_environment_variables():
    """Initialize environment variables for OpenAI and LangChain tracking."""
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['LANGCHAIN_TRACING_V2'] = "true"
    os.environ['LANGCHAIN_PROJECT'] = 'Q&A Chatbot with OpenAI'

# Function to create the OpenAI LLM model
def create_openai_model(api_key, model_name, temperature, max_tokens):
    """
    Create and configure the OpenAI model based on user input.
    
    Args:
        api_key (str): OpenAI API key for authentication.
        model_name (str): The model name to use, e.g., 'gpt-4'.
        temperature (float): Controls the randomness of the model's output.
        max_tokens (int): Maximum number of tokens for response generation.
        
    Returns:
        ChatOpenAI: Configured OpenAI language model instance.
    """
    # Check if the API key is valid
    if not api_key:
        st.warning("Please provide a valid OpenAI API key.")
        return None
    try:
        # Initialize OpenAI model with provided parameters
        llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens, openai_api_key=api_key)
        return llm
    except Exception as e:
        st.error(f"Error initializing OpenAI model: {e}")
        return None

# Function to generate the response from the LLM model
def generate_response(question, llm, prompt):
    """
    Generate a response from the LLM model based on the input question.
    
    Args:
        question (str): User's query to the chatbot.
        llm (ChatOpenAI): Configured LLM instance.
        prompt (ChatPromptTemplate): The prompt template to use for the LLM.
        
    Returns:
        str: Generated response from the LLM model.
    """
    try:
        # Create Output Parser
        output_parser = StrOutputParser()

        # Build the chain: prompt | llm | output_parser
        chain = prompt | llm | output_parser

        # Invoke the chain and return the response
        answer = chain.invoke({'Question': question})
        return answer
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Failed to generate response."

# Function to build the Streamlit UI components
def build_ui():
    """
    Build the Streamlit UI components for user input and displaying the chatbot response.
    """
    # Title of the app
    st.title("Enhanced Q&A Chatbot with OpenAI")

    # Sidebar settings for OpenAI API key
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

    # Dropdown to select various models
    llm_model = st.sidebar.selectbox("Select an OpenAI model", ["gpt-4", "gpt-4-turbo"])

    # Adjust response parameters
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

    return api_key, llm_model, temperature, max_tokens

# Function to run the Streamlit app
def run_app():
    """
    Run the Streamlit app, collect user input, and display responses from the chatbot.
    """
    # Initialize environment variables
    initialize_environment_variables()

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user queries."),
            ("user", "Question: {Question}")
        ]
    )

    # Build the Streamlit UI and get user inputs
    api_key, llm_model, temperature, max_tokens = build_ui()

    # Main interface for user input
    st.write("Go ahead & ask any question")
    user_input = st.text_input('You:')

    # Check if both API key and user input are provided
    if api_key and user_input:
        # Create OpenAI model instance
        llm = create_openai_model(api_key, llm_model, temperature, max_tokens)
        if llm:
            # Generate and display response from the LLM
            response = generate_response(user_input, llm, prompt)
            st.write(response)
    elif not api_key:
        st.warning("Please provide your OpenAI API key.")
    elif not user_input:
        st.info("Please ask a question.")

# Run the Streamlit app
if __name__ == "__main__":
    run_app()
