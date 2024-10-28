import openai 
import streamlit as st
import tiktoken
import json
import time

import plotly.express as px
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from utils.document_processing import preprocess_text_for_bm25


# Query modifier for sql query generator
modified_user_query_template = """
### User Query: 
{__user_query__}

### The following context contains sample rows from the data to help you construct the correct SQL query. Use this context to identify the appropriate columns and create an SQL query that accurately retrieves the required information.
```
{__context__}
```
"""

def summary_generation(pages, api_key, model):
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0, model_name=model)
    chain = load_summarize_chain(llm, chain_type="stuff")

    result = chain.invoke(pages)

    output_words = len(result["output_text"].split())
    output_tokens = num_tokens_from_string(result["output_text"], encoding_name="cl100k_base")

    return result["output_text"], output_words, output_tokens

def response_generation_using_file_search(
    prompt      
):
    
    # Storing Client, Assistant, File-Batch
    client = st.session_state["client"]
    file_batch = st.session_state["file_batch"]
    assistant = st.session_state["assistant"]

    # Create a thread
    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": prompt}]
    )

    # Initiate Process
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    
    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []

    # Add Citation Information
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, "")
    
    response = message_content.value

    output_words = len(message_content.value.split())
    output_tokens = num_tokens_from_string(message_content.value, encoding_name="cl100k_base")

    return response, output_words, output_tokens


def chat_completion(
    api_key=None, model="", prompt="", temperature=0, max_tokens=1000,
):
    """
    Description: Function to call the OpenAI API.\n
    Inputs
    -------
        - api_key: Openai API Key (default is None)
        - models: the model to be used for generating text
        - prompt: the prompt to be used for generating text
        - temperature: the temperature to be used for generating text
    Outputs
    -------
        - text: the generated text
        - tokens: the number of tokens used for generating text
        - words: the number of words used for generating text
    """

    # Initialize Client
    client = openai.OpenAI(
        api_key=api_key,
        timeout=None
    )
    for i in range(2):
        try:
            llm_response = client.chat.completions.create(
                model=model, 
                messages=[{
                    "role": "user", 
                    "content": prompt
                }], 
                temperature=temperature,
                seed=1234,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            response = llm_response.choices[0].message.content
            break
        except:    
            response = '{"response": [{"type": "text", "content": "Oops! It looks like our AI assistant is taking an unexpected break. We\'re having a bit of trouble generating a response right now. This could be due to temporary issues with our AI service or high demand. You can wait a moment and try your request again. If the problem persists, please try again later or contact our support team. We apologize for the inconvenience and appreciate your patience!"}]}'
    print(response)
    
    # get the reason for stopping the text generation
    output = json.loads(response)
    # print(output)
    try:
        tokens = llm_response.usage.completion_tokens
    except:
        tokens = -1
    words = len(response.split())

    return output, tokens, words

def chatbot(
    api_key=None, model="", prompt=""
):
    # Initialize Client
    client = openai.OpenAI(
        api_key=api_key,
        timeout=None
    )
    
    if user_input := st.chat_input("Ask here!"):
        # Re-initialize emtpy chat (Currently, we are not storing history)
        st.session_state.messages = []

        # Add User Query into the message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show user query
        with st.chat_message("user"):
            st.write(user_input)

        # Generate Response and show
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": m["role"], "content": prompt.format(__texts__=st.session_state["extracted_text"], __user_query__=m["content"])} for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Show last QA pair
        if "messages" in st.session_state:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    return

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """
    Description: Function to count number of tokens in a text string
        - num_tokens_from_string function to count number of tokens in a text string
        - uses tiktoken to count number of tokens in a text string
        - parameters: "string" is the text string, "encoding_name" is the encoding name to be used by tiktoken
        - returns: num_tokens->number of tokens in the text string
        - This function is used within extract_data, extract_page, extract_YT, extract_audio, extract_image functions
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def key_validation(input_key):
    openai.OpenAI(api_key=input_key).chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=2,
        temperature=0,
        seed=1234
    )
    return

def text_response(response, api_key, model, stream):
    # Initialize Client
    client = openai.OpenAI(
        api_key=api_key,
        timeout=None
    )

    response = client.chat.completions.create(
        model=model, 
        messages=[{
            "role": "user", 
            "content": f"Please convert the following JSON into a textual response:\n```{response}```"
        }], 
        temperature=0,
        seed=1234
    )

    # get the reason for stopping the text generation
    output = f"Encountered an error while generating plots - Generating only text output\n\n"+response.choices[0].message.content

    if stream:
        st.write_stream(stream_data(output))
    else:
        st.markdown(output.replace(r"$", "&#36;"), unsafe_allow_html=True)
    return 

def stream_data(chunk):
    print(chunk)
    for i in range(len(chunk["content"].split())):
        yield chunk["content"].split()[i].replace(r"$", "&#36;") + " "
        time.sleep(0.02)
                
def response_with_plots(response, stream):
    for chunk in response['response']:
        if chunk['type'] == 'data':
            plot_type = chunk['content']['plot_type']
            data = chunk['content']['data']
            layout = chunk['content']['layout']

            if plot_type == 'bar':
                # Check if y data is a list of dictionaries (grouped data)
                if isinstance(data['y'][0], dict):
                    df = []
                    for index, category in enumerate(data['x']):
                        if index < len(data['y']):
                            for group, value in data['y'][index].items():
                                df.append({'category': category, 'group': group, 'value': value})
                        else:
                            raise IndexError(f"Index {index} out of range for y data.")
                    fig = px.bar(pd.DataFrame(df), x='category', y='value', color='group', barmode='group')
                else:
                    fig = px.bar(x=data['x'], y=data['y'])
                    
            elif plot_type == 'line':
                # If y is a dictionary, create a DataFrame with multiple series
                if isinstance(data['y'], dict):
                    df = pd.DataFrame({'x': data['x'], **data['y']})
                    fig = px.line(df, x='x', y=list(data['y'].keys()))
                else:
                    fig = px.line(x=data['x'], y=data['y'])
                    
            elif plot_type == 'scatter':
                fig = px.scatter(x=data['x'], y=data['y'])
            
            # Add layout customizations
            fig.update_layout(
                title=layout.get('title', ''),
                xaxis_title=layout.get('xaxis', {}).get('title', ''),
                yaxis_title=layout.get('yaxis', {}).get('title', '')
            )
            # temp = st.container(height=600, border=False)
            st.plotly_chart(fig, use_container_width=False)
        else:
            if stream:
                st.write_stream(stream_data(chunk))
            else:
                st.markdown(chunk["content"].replace(r"$", "&#36;"), unsafe_allow_html=True)

def response_for_csv(user_query, dbs, api_key, model, data):
    """Generate response when csv file is uploaded in the tool."""
    sql_db, bm25 = dbs
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=api_key, temperature=0, max_tokens=4096)

    # Initialize SQL Agent
    agent_executor = create_sql_agent(
        llm, db=sql_db, agent_type="zero-shot-react-description",
        max_iterations=10, max_execution_time=500, top_k=10, verbose=True
    )

    # Query
    tokenized_query = preprocess_text_for_bm25(user_query)
    print(tokenized_query)

    # Perform search to filter relevant rows
    doc_scores = bm25.get_scores(tokenized_query)

    # Sorted results
    results = [(i, score) for i, score in enumerate(doc_scores)]
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Context to add in user query
    context = ""
    for res_ in sorted_results[:10]:
        if res_[1] > 0:
            context += f"---\n{data[res_[0]].page_content.replace("{", "[").replace("}", "]")}\n"
    print(context)
    
    if context != "":
        updated_query = modified_user_query_template.format(__user_query__=user_query, __context__=context)
        # Generate response
        result = agent_executor.invoke({"input": updated_query})
    else:        
        # Generate response
        result = agent_executor.invoke({"input": user_query})

    return result["output"]

            
