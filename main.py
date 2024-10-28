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
    """Load prompt files with detailed error handling and logging"""
    import traceback
    
    prompt_files = {
        'summary_prompt': "prompts/summary_prompt.txt",
        'qna_instructions_example': "prompts/qna_instructions_example.txt",
        'qna_input': "prompts/qna_input.txt",
        'chunk_template': "prompts/chunk_template.txt"
    }
    
    results = {}
    
    for key, filepath in prompt_files.items():
        st.write(f"Attempting to read {filepath}")  # Debug log
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']
        success = False
        
        for encoding in encodings:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    content = f.read()
                    results[key] = content
                    st.write(f"Successfully read {filepath} with {encoding} encoding")  # Debug log
                    success = True
                    break
            except UnicodeDecodeError as e:
                st.write(f"Failed to read {filepath} with {encoding} encoding: {str(e)}")  # Debug log
                continue
            except FileNotFoundError:
                st.error(f"File not found: {filepath}")
                raise
            except Exception as e:
                st.error(f"Unexpected error reading {filepath}: {str(e)}")
                st.write(traceback.format_exc())  # Full traceback
                raise
        
        if not success:
            st.error(f"Failed to read {filepath} with any encoding")
            raise ValueError(f"Could not read {filepath} with any supported encoding")
    
    return (results['summary_prompt'], results['qna_instructions_example'], 
            results['qna_input'], results['chunk_template'])

@st.cache_data(persist=True)
def load_config():
    """Loads configuration from config.ini file with explicit encoding"""
    config_object = ConfigParser()
    try:
        config_object.read("config.ini", encoding='utf-8')
    except UnicodeDecodeError:
        config_object.read("config.ini", encoding='latin-1')

    # Load Objects
    logo = Image.open(config_object["IMAGES"]["logo_address"]) 
    llm_model = config_object["OPENAI LANGUAGE MODELS"]["model3"]
    emb_model = config_object["OPENAI EMBEDDING MODELS"]["model1"]
    
    # Get all messages from the "Messages" section
    msgs = {key: config_object["MESSAGES"][key] for key in config_object["MESSAGES"]}

    return logo, llm_model, emb_model, msgs

@st.cache_data(persist=True)
def main_page_header(_logo):
    """Render main page header."""
    col1, _ = st.columns([1, 7])
    col1.image(_logo)
    st.markdown("###### Demo Tool")

def main():
    """Main function to run the streamlit application."""
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
        widgets.check_key()
        input_choice, file = widgets.data_upload()

        if file is not None and "db" not in st.session_state:
            with st.spinner("Processing Documents"):
                file_type, pages, db, embedding_fn = document_processing.process_document(
                    input_choice, file, st.session_state["openai_api_key"])
                st.session_state.update({
                    "file_type": file_type,
                    "pages": pages,
                    "db": db,
                    "embedding_fn": embedding_fn
                })
                st.rerun()

    with tab2:
        if "db" in st.session_state:
            if st.session_state["file_type"] == "csv":
                st.markdown("At present, the tool supports summary generation only for PDF files.")
            else:
                if "summary" not in st.session_state and st.button("Click here to generate summary"):
                    with st.spinner("Crafting your summary ..."):
                        response, output_words, output_token = response_generation.summary_generation(
                            st.session_state["pages"], 
                            st.session_state["openai_api_key"], 
                            llm_model
                        )

                        st.session_state.update({
                            "summary": response.replace("$", "&#36;"),
                            "output_words": output_words,
                            "output_tokens": output_token
                        })
                        
                        st.markdown(st.session_state["summary"], unsafe_allow_html=True)
                        st.markdown(f"**:blue[Words]**: :grey[{st.session_state['output_words']}], "
                                  f"**:blue[Tokens]**: :grey[{st.session_state['output_tokens']}]")
                        st.rerun()

            if "summary" in st.session_state:
                st.markdown(st.session_state["summary"], unsafe_allow_html=True)
                st.markdown(f"**:blue[Words]**: :grey[{st.session_state['output_words']}], "
                          f"**:blue[Tokens]**: :grey[{st.session_state['output_tokens']}]")
        else:
            st.markdown("Kindly upload the necessary files to utilize this page.")

    with tab3:
        if "db" in st.session_state:
            if user_input := st.chat_input("Ask here!", key="user_input"):
                st.session_state.messages = []
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Crafting Response"):       
                        if st.session_state["file_type"] == "csv":
                            context = response_generation.response_for_csv(
                                user_input, st.session_state["db"], 
                                st.session_state["openai_api_key"], 
                                llm_model, st.session_state["pages"]
                            )
                        else:
                            user_query_vectors = st.session_state["embedding_fn"].embed_documents([user_input])
                            filtered_results = st.session_state["db"].search(
                                collection_name="demo_collection",        
                                data=user_query_vectors,               
                                limit=5,                           
                                output_fields=["text", "source", "page"],
                            )

                            context = "".join(
                                chunk_template.format(
                                    page=chunk["entity"]["page"]+1, 
                                    content=chunk["entity"]["text"]
                                ) for chunk in filtered_results[0]
                            )
                            
                        st.session_state["prompt"] = (qna_instructions_example + 
                            qna_input.format(__texts__=context, __user_query__=user_input))
                            
                        response, _, _ = response_generation.chat_completion(
                            api_key=st.session_state["openai_api_key"],
                            model=llm_model, 
                            prompt=st.session_state["prompt"],
                            max_tokens=1000,
                        )   

                    try:
                        response_generation.response_with_plots(response, True)
                    except:
                        st.rerun()
                        st.markdown("Encountered an error while generating plots - Generating only text output")
                        response_generation.text_response(response, st.session_state["openai_api_key"], 
                                                       llm_model, False)

                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                if "messages" in st.session_state:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            if message["role"] == "assistant":
                                try:
                                    response_generation.response_with_plots(message["content"], False)
                                except:
                                    response_generation.text_response(message["content"], 
                                                                   st.session_state["openai_api_key"], 
                                                                   llm_model, False)
                            else:
                                st.markdown(message["content"], unsafe_allow_html=False)
        else:
            st.markdown("Kindly upload the necessary files to utilize this page.")

if __name__ == "__main__":
    main()
