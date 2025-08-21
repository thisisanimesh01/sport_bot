

import streamlit as st
from src.sports_chatbot import SportsChatbot
import time

# Page Configuration in the Streamlit app
st.set_page_config(
    page_title="Sports Intelligence Chatbot",
    page_icon="⚽",
    layout="wide"
)


st.title("🏆 Sports Intelligence Chatbot")    # User-facing title
st.markdown("""
Welcome to the Sports Intelligence Chatbot! Ask me anything about your chosen sport.
This chatbot uses a Retrieval-Augmented Generation (RAG) system to answer questions based on a knowledge base.
""")

# caching the Chatbot Initialization 
# This is crucial for performance. It prevents reloading the models on every user interaction.
@st.cache_resource
def load_chatbot():
    """Loads and caches the chatbot instance."""
    with st.spinner("Initializing the chatbot... This may take a moment."):
        try:
            bot = SportsChatbot()
            return bot
        except Exception as e:
            st.error(f"Failed to initialize the chatbot: {e}", icon="🚨")
            return None

# Load the chatbot from the cache
bot = load_chatbot()

if bot:
    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you with your sports questions today?"}]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about sports..."):  # User input
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})  # User message
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    full_response = bot.answer(prompt)
                    # Simulate a streaming effect for better UX
                    response_text = ""
                    for chunk in full_response.split():
                        response_text += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(response_text + "▌")  # Simulate streaming
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"Sorry, an error occurred: {e}"
                    st.error(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.warning("Chatbot is not available due to an initialization error. Please check the logs.")    # Log initialization errors

 #this is the demo.py file where all the Streamlit UI code resides and also it's an entry point for the application