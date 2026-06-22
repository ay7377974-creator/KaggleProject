"""
Zyro Dynamics HR Help Desk — Streamlit Chatbot
================================================
Deploy this file as app.py on Streamlit Community Cloud
(https://share.streamlit.io).

Required secrets (Settings -> Secrets on Streamlit Cloud):
    GROQ_API_KEY = "your-groq-key"

The HR policy PDFs must be present in a folder named `hr_docs/` alongside
this app.py in your GitHub repo (see deployment instructions).
"""

import os
import glob
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="💼", layout="centered")

CORPUS_PATH = "hr_docs"
LLM_MODEL = "llama-3.1-8b-instant"

REFUSAL_MESSAGE = (
    "I can only answer HR-related questions based on this company's internal "
    "policy documents. This question falls outside that scope, so I'm not "
    "able to answer it. Please reach out to the relevant team or your manager "
    "for help with this."
)

OOS_KEYWORDS = [
    "weather", "stock price", "stock market", "revenue", "recruitment",
    "hiring process", "apply for a job", "job application", "esop",
    "stock option", "product feature", "salesforce", "competitor",
    "zoho", "freshworks", "cricket score", "football score", "recipe",
    "capital of", "president of", "prime minister", "write code",
    "write a poem", "translate this", "joke", "bitcoin", "cryptocurrency",
]

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the HR Help Desk assistant for this company. Answer the "
     "employee's question using ONLY the information in the CONTEXT below, "
     "which is retrieved from the official HR policy documents.\n\n"
     "Rules:\n"
     "1. Base your answer strictly on the provided context. Do not use outside knowledge.\n"
     "2. If the context does not contain enough information to answer the question, "
     "say clearly that the policy documents do not cover this and you cannot answer it.\n"
     "3. Be concise, clear, and professional.\n"
     "4. Where relevant, cite specific numbers, durations, or policy names found in the context.\n"
     "5. Do not invent policy details that are not explicitly stated in the context.\n"
     "6. The documents may refer to the company by more than one name (e.g. due to a legal "
     "rebrand) -- treat all such names in the context as referring to the same company.\n\n"
     "CONTEXT:\n{context}"),
    ("human", "Employee question: {question}"),
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a classifier for an HR Help Desk bot. Decide if the employee's "
     "question can plausibly be answered using INTERNAL HR POLICY DOCUMENTS "
     "(topics like: leave, payroll, benefits, performance reviews, WFH, code "
     "of conduct, IT/security policy, POSH, onboarding/separation, travel & "
     "expense). Questions about hiring/recruitment process for external "
     "candidates, company financials/revenue, product features, stock "
     "options/ESOP details, or policies of OTHER companies are OUT OF SCOPE. "
     "Reply with exactly one word: INSCOPE or OUTOFSCOPE."),
    ("human", "{question}"),
])


# ---------------------------------------------------------------------------
# Pipeline (cached so it only builds once per app session/server)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not groq_key:
        st.error("GROQ_API_KEY not found. Add it under Settings -> Secrets in Streamlit Cloud.")
        st.stop()

    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
    )

    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=512, groq_api_key=groq_key)

    def format_docs(docs):
        formatted = []
        for d in docs:
            src = d.metadata.get("source", "unknown").split("/")[-1].split("\\")[-1]
            page = d.metadata.get("page", "?")
            formatted.append(f"[Source: {src}, page {page}]\n{d.page_content}")
        return "\n\n---\n\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )

    oos_classifier = OOS_PROMPT | llm | StrOutputParser()

    return retriever, rag_chain, oos_classifier


def looks_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in OOS_KEYWORDS)


def ask_bot(question: str, retriever, rag_chain, oos_classifier):
    if looks_out_of_scope(question):
        return {"answer": REFUSAL_MESSAGE, "sources": [], "refused": True}

    try:
        verdict = oos_classifier.invoke({"question": question}).strip().upper()
    except Exception:
        verdict = "INSCOPE"

    if "OUTOFSCOPE" in verdict:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "refused": True}

    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    sources, seen = [], set()
    for d in docs:
        src = d.metadata.get("source", "unknown").split("/")[-1].split("\\")[-1]
        page = d.metadata.get("page", "?")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source_file": src, "page": page})

    return {"answer": answer, "sources": sources, "refused": False}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("💼 Zyro Dynamics HR Help Desk")
st.caption("Ask me about leave, payroll, benefits, WFH, performance reviews, and other HR policies.")

if not os.path.isdir(CORPUS_PATH) or not glob.glob(os.path.join(CORPUS_PATH, "*.pdf")):
    st.error(
        f"No PDF files found in `{CORPUS_PATH}/`. Make sure the HR policy PDFs are "
        f"committed to your repo inside a folder named `{CORPUS_PATH}/`."
    )
    st.stop()

with st.spinner("Loading HR policy documents and building the knowledge base..."):
    retriever, rag_chain, oos_classifier = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm the Zyro Dynamics HR Help Desk bot. What would you like to know?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source_file']}**, page {s['page']}")

user_question = st.chat_input("Ask an HR question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_bot(user_question, retriever, rag_chain, oos_classifier)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📄 Sources"):
                for s in result["sources"]:
                    st.markdown(f"- **{s['source_file']}**, page {s['page']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot answers HR policy questions using Retrieval-Augmented "
        "Generation (RAG) over Zyro Dynamics' internal HR documents. "
        "Out-of-scope questions (anything not covered by HR policy) are "
        "politely declined."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm the Zyro Dynamics HR Help Desk bot. What would you like to know?"}
        ]
        st.rerun()
