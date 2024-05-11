import streamlit as st
from langchain_utils import invoke_chain
from unsloth import FastLanguageModel

st.title("Langchain NL2SQL Chatbot")

from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
import torch
from context_retriever import ContextRetriever

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# Set a default model
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("basavaraj/text2sql-Llama3-8b")
    model = LlamaForCausalLM.from_pretrained(
    "basavaraj/text2sql-Llama3-8b",
    load_in_4bit=True,
    torch_dtype=torch.float16,)
    FastLanguageModel.for_inference(model)
    contextRetriever = ContextRetriever()
    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer
    st.session_state["contextRetriever"] = contextRetriever

if "model" not in st.session_state:
    st.session_state["model_name"] = "basavaraj/text2sql-Llama3-8b"
    load_model()

# Initialize chat history
if 'messages' not in st.session_state:
    print("Creating session state")
    st.session_state.messages = []

print("st.session_state.messages :", st.session_state.messages)
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            response = invoke_chain(prompt, st.session_state.messages, st.session_state.tokenizer, st.session_state.model, st.session_state.contextRetriever)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
