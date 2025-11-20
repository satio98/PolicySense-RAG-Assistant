from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from setup_index import load_embedding
from prompt_template import load_prompt
from flask import Flask, render_template, request
import re

app = Flask(__name__)

# Set up vectorstore
embedding = None
index_name = 'legal'

if not embedding:
    embedding = load_embedding()

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# Build retriever
retriever = vectorstore.as_retriever(
    search_type = 'similarity',
    search_kwargs={"k": 10}
    )

# Load OpenAI or another LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Build RAG pipeline
prompt = load_prompt()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

@app.route("/")
def index():
    return render_template('chat.html')

def markdown_to_html(text: str):
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

def format_llm_output(raw):
    html = markdown_to_html(raw)

    # Add two newlines after Answer:
    html = html.replace("<b>Answer:</b>", "<b>Answer:</b><br>")

    # Add spacing before Supported By:
    html = html.replace("<b>Supported By:</b>", "<br><br><b>Supported By:</b><br>")

    return html

@app.route("/get", methods=['GET', 'POST'])
def chat():
    message = request.form['msg']
    result = qa.invoke({"query": message})

    raw_answer = result["result"]
    answer = format_llm_output(raw_answer)

    # Build source list (short version)
    doc = result.get("source_documents", [])[0]
    md = doc.metadata
    page = int(md.get("page", 0)) + 1
    total = md.get("total_pages")
    text = f"•Filename: {md['source'][5:]}<br>•Page: {page}{f' of {int(total)}' if total else ''}"

    if text:
        full_response = (
            f"{answer}"
            f"<br><br><b>Source(s):</b><br>{text}"
        )
    else:
        full_response = answer

    return full_response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
