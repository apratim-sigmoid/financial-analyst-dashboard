import streamlit as st
import utils.response_generation as response_generation
import time

st.cache(persist=True)
def data_upload():
    """
    Description: Function to display document input options and return the input choice and uploaded file
    """

    input_choice = st.radio(
        label="###### :blue[Please upload the file to get started]", options=("Document", "File Directory"), 
        disabled="openai_api_key" not in st.session_state

    )

    if input_choice == "Document":
        with st.expander("📁 __Documents__", expanded=True):
            uploaded = st.file_uploader(
                label="Select File", type=['pdf', 'csv'], on_change=clear,
                disabled="openai_api_key" not in st.session_state
            )
    # Disabling multiple file uploads
    # elif input_choice == "File Directory":
    #     with st.expander("📁 __File Directory__", expanded=True):
    #         uploaded = st.file_uploader(
    #             label="Select File", type=['pdf'], on_change=clear, 
    #             accept_multiple_files=True,
    #             disabled="openai_api_key" not in st.session_state
    #         )
            
    #         if uploaded == []:
    #              uploaded = None
    else:
        uploaded = None

    return input_choice, uploaded

st.cache(persist=True)
def check_key():
    col, _, = st.columns([2, 4])
    if input_key := col.text_input(
        "###### :blue[Please provide OpenAI API Key]", max_chars=200, type="password",
        key="openai_input_key"
    ):
        
        try:
            response_generation.key_validation(input_key)
            st.session_state["openai_api_key"] = input_key
            col.success("Valid Key!")
        except:
            col.error("Invalid Key!")

def clear():
    """
    Description: Function to clear the cache and initialize the chat
    """
    with st.spinner("Clearing all history..."):
        st.cache_data.clear()
        
        for key in st.session_state.keys():
            if key != "openai_api_key":
                del st.session_state[key]
