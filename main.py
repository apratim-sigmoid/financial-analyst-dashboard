import streamlit as st
import streamlit_authenticator as stauth
from configparser import ConfigParser
from PIL import Image
import json

# Custom Modules
import utils.widgets as widgets
import utils.document_processing as document_processing
import utils.response_generation as response_generation

@st.cache_data(persist=True)
def load_prompts():
    # Summary Prompt
    with open("prompts/summary_prompt.txt", "r") as f:
        summary_prompt = f.read()

    f.close()

    # Question-Answering Prompt
    with open("prompts/qna_instructions_example.txt", "r") as f:
        qna_instructions_example = f.read()

    f.close()

    # Question-Answering Prompt
    with open("prompts/qna_input.txt", "r") as f:
        qna_input = f.read()

    f.close()

    # Question-Answering Prompt
    with open("prompts/chunk_template.txt", "r") as f:
        chunk_template = f.read()

    f.close()

    return summary_prompt, qna_instructions_example, qna_input, chunk_template

@st.cache_data(persist=True)
def load_config():
    """Loads configuration from config.ini file"""
    # Initialize & Read configuration files
    config_object = ConfigParser()
    config_object.read("config.ini")


    # Load Objects
    logo = Image.open(config_object["IMAGES"]["logo_address"]) 
    llm_model = config_object["OPENAI LANGUAGE MODELS"]["model3"]
    emb_model = config_object["OPENAI EMBEDDING MODELS"]["model1"]
    
    # Get all messages from the "Messages" section
    msgs = {key: config_object["MESSAGES"][key] for key in config_object["MESSAGES"]}

    return logo, llm_model, emb_model, msgs

@st.cache_data(persist=True)
def main_page_header(_logo):
    """
    Description: Render main page header.
    """
    col1, _ = st.columns([1, 7])

    col1.image(_logo)
    st.markdown("###### Demo Tool")
    
    return

def main():
    """
    Description: Main function to run the streamlit application.
    """
    st.set_page_config(
        page_title="Demo Tool",
        layout="wide",
    )

    # Load configuration (cached)
    logo, llm_model, _, msgs = load_config()

    # Load Prompts
    summary_prompt, qna_instructions_example, qna_input, chunk_template = load_prompts()

    # Render Header Title & Logo
    main_page_header(logo)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Introduction", "Summary", "Chat"])

    with tab1:
        st.markdown(msgs["introduction"])
                
        # Check key uploaded by the user
        widgets.check_key()
        
        # Ask user to upload the file(s)
        input_choice, file = widgets.data_upload()

        # Parse PDF ----------------------------------------------------------------------------------------------
        if file is not None and "db" not in st.session_state:
            with st.spinner("Processing Documents"):
                file_type, pages, db, embedding_fn = document_processing.process_document(input_choice, file, st.session_state["openai_api_key"])
                st.session_state["file_type"] = file_type
                st.session_state["pages"] = pages
                st.session_state["db"] = db
                st.session_state["embedding_fn"] = embedding_fn
                st.rerun()
        # --------------------------------------------------------------------------------------------------------

    with tab2:
        # Generate response using Chat Completion API ------------------------------------------------------------
        if "db" in st.session_state:
            if st.session_state["file_type"] == "csv":
                st.markdown("At present, the tool supports summary generation only for PDF files.")
            else:
                if "summary" not in st.session_state and st.button("Click here to generate summary"):
                    with st.spinner("Crafting your summary ..."):
                        # # Pass whole content
                        # context = ""
                        # for i in range(len(st.session_state["pages"])):
                        #     context += f"### Page {st.session_state["pages"][i].metadata["page"]}\n\n" + st.session_state["pages"][i].page_content + "\n\n"

                        # # Generate response
                        # response, output_words, output_tokens = response_generation.chat_completion(
                        #     api_key=st.session_state["openai_api_key"], model=llm_model, prompt=summary_prompt.format(__texts__=context), max_tokens=4000
                        # )
                        response, output_words, output_token = response_generation.summary_generation(st.session_state["pages"], st.session_state["openai_api_key"], llm_model)

                        # Storing Output
                        st.session_state["summary"] = response.replace("$", "&#36;")
                        st.session_state["output_words"] = output_words
                        st.session_state["output_tokens"] = output_token
                        
                        # try:
                        #     response_generation.response_with_plots(response, True)
                        # except:
                        #     response_generation.text_response(response, st.session_state["openai_api_key"], llm_model, False)
                        st.markdown(st.session_state["summary"], unsafe_allow_html=True)
                        st.markdown(f"**:blue[Words]**: :grey[{st.session_state["output_words"]}], **:blue[Tokens]**: :grey[{st.session_state["output_tokens"]}]")
                        st.rerun()


            if "summary" in st.session_state:
                # try:
                #     response_generation.response_with_plots(st.session_state["summary"], False)
                # except:
                #     st.markdown(f"Encountered an error while generating plots - Generating only text output")
                #     response_generation.text_response(st.session_state["summary"], st.session_state["openai_api_key"], llm_model, False)
                st.markdown(st.session_state["summary"], unsafe_allow_html=True)
                st.markdown(f"**:blue[Words]**: :grey[{st.session_state["output_words"]}], **:blue[Tokens]**: :grey[{st.session_state["output_tokens"]}]")
        # --------------------------------------------------------------------------------------------------------

        else:
            st.markdown("Kindly upload the necessary files to utilize this page.")

    with tab3:
        # Generate response using Openai file-search
        if "db" in st.session_state:
            if user_input := st.chat_input("Ask here!", key="user_input"):
                # Re-initialize emtpy chat (Currently, we are not storing history)
                st.session_state.messages = []

                # Add User Query into the message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Show user query
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Generate Response and show
                with st.chat_message("assistant"):
                    with st.spinner("Crafting Response"):       
                        
                        # Chat Completion API --------------------------------------------------------------------

                        if st.session_state["file_type"] == "csv":
                            # Generate response for CSV
                            context = response_generation.response_for_csv(
                                user_input, st.session_state["db"], st.session_state["openai_api_key"], llm_model, st.session_state["pages"]
                            )
                            print(">"*3, "response from agent", context)
                        else:
                            # Filter Context
                            user_query_vectors = st.session_state["embedding_fn"].embed_documents([user_input])

                            filtered_results = st.session_state["db"].search(
                                collection_name="demo_collection",        
                                data=user_query_vectors,               
                                limit=5,                           
                                output_fields=["text", "source", "page"],
                            )

                            # Prepare Input Context
                            context = ""
                            for chunk in filtered_results[0]:
                                context += chunk_template.format(page=chunk["entity"]["page"]+1, content=chunk["entity"]["text"])
                            
                        st.session_state["prompt"] = qna_instructions_example + qna_input.format(__texts__=context, __user_query__=user_input)
                            
                        # Generate Response
                        response, _, _= response_generation.chat_completion(
                            api_key=st.session_state["openai_api_key"], model=llm_model, 
                            prompt=st.session_state["prompt"],
                            max_tokens=1000,
                        )   
                        #-----------------------------------------------------------------------------------------

                      
                    # Save Input & Output & Retrieval Results
                    # f = open("logs/input.txt", "w")
                    # f.write(st.session_state["prompt"])
                    # f.close()

                    # f = open("logs/output.txt", "w")
                    # f.write(json.dumps(response))
                    # f.close()

                    try:
                        response_generation.response_with_plots(response, True)
                    except:
                        st.rerun()
                        st.markdown(f"Encountered an error while generating plots - Generating only text output")
                        response_generation.text_response(response, st.session_state["openai_api_key"], llm_model, False)

                    # Store Response
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Show last QA pair
                if "messages" in st.session_state:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            if message["role"] == "assistant":
                                try:
                                    response_generation.response_with_plots(message["content"], False)
                                except:
                                    response_generation.text_response(message["content"], st.session_state["openai_api_key"], llm_model, False)
                            else:
                                st.markdown(message["content"], unsafe_allow_html=False)
                        
                            
        else:
            st.markdown("Kindly upload the necessary files to utilize this page.")

    return

if __name__ == "__main__":
  main()
