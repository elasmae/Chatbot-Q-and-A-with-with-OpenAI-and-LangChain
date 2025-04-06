# ✅ Must be first Streamlit call
import streamlit as st
st.set_page_config(page_title="GPT-2 PDF Chatbot", layout="wide")

# ✅ Optional: clean noisy warnings from torch etc.
import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import os

# --- Load GPT-2 model ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, truncation=True)
    return pipe, tokenizer

pipe, tokenizer = load_model()

# --- Custom embedding wrapper ---
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts): return self.model.encode(texts, convert_to_tensor=False).tolist()
    def embed_query(self, text): return self.model.encode(text, convert_to_tensor=False).tolist()

# --- Load vector DB and chain ---
@st.cache_resource
def load_chain():
    loader = PyPDFLoader("Introduction_to_Tableau.pdf")
    docs = loader.load()

    splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(docs)

    embedding = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs_split, embedding, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda": 0.7})

    prompt_creating_question = PromptTemplate.from_template('''Lecture: {question_lecture}
Title: {question_title}
Body: {question_body}''')

    prompt_retrieving_s = SystemMessage(content='''You will receive a question from a student taking a Tableau course, which includes a title and a body. 
The corresponding lecture will also be provided.

Answer the question using only the provided context.

At the end of your response, include the section and lecture names where the context was drawn from, formatted as follows: 
Resources: 
Section: *Section Title*, Lecture: *Lecture Title*''')

    prompt_template_retrieving_h = HumanMessagePromptTemplate.from_template('''This is the question:
{question}

This is the context:
{context}''')

    chat_prompt_template_retrieving = ChatPromptTemplate(
        messages=[prompt_retrieving_s, prompt_template_retrieving_h]
    )

    def format_context(dictionary):
        formatted_string = ""
        for i, doc in enumerate(dictionary["context"]):
            section = doc.metadata.get("section_title", "Unknown Section")
            lecture = doc.metadata.get("lecture_title", "Unknown Lecture")
            formatted_string += f"""
Document {i + 1}
Section Title: {section}
Lecture Title: {lecture}
Content: {doc.page_content.strip()}

-------------------
"""
        dictionary["context"] = formatted_string
        return dictionary

    def truncate_prompt(prompt: str, max_tokens: int = 900):
        tokens = tokenizer.encode(prompt)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return tokenizer.decode(tokens, skip_special_tokens=True)
        return prompt

    def format_response_text(text: str) -> str:
        return text.replace("\\n", "\n").replace("  ", " ").strip()

    from langchain.llms import HuggingFacePipeline

    get_text = RunnableLambda(lambda x: x.to_string())
    combine_with_context = RunnableLambda(lambda q: {
        "context": retriever.invoke(q),
        "question": q
    })
    apply_truncation = RunnableLambda(lambda d: {
        "question": d["question"],
        "context": truncate_prompt(d["context"])
    })
    final_prompt_truncation = RunnableLambda(lambda messages: truncate_prompt(str(messages)))

    chain = (
        prompt_creating_question
        | get_text
        | combine_with_context
        | RunnableLambda(format_context)
        | apply_truncation
        | chat_prompt_template_retrieving
        | final_prompt_truncation
        | HuggingFacePipeline(pipeline=pipe)
    )

    return chain, format_response_text

# --- Streamlit UI ---
st.title("📘 Ask Questions About Tableau (PDF Q&A)")
st.markdown("Type a question below based on a lecture from the PDF course.")

question_lecture = st.text_input("Lecture Name")
question_title = st.text_input("Question Title")
question_body = st.text_area("Question Body")

if st.button("Ask"):
    if not all([question_lecture, question_title, question_body]):
        st.warning("Please fill in all fields.")
    else:
        chain, formatter = load_chain()
        question_input = {
            "question_lecture": question_lecture,
            "question_title": question_title,
            "question_body": question_body
        }
        result = chain.invoke(question_input)
        formatted = formatter(str(result))
        st.markdown("### 💬 Answer")
        st.code(formatted)
