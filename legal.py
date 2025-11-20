from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from setup_index import load_embedding
from prompt_template import load_prompt

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

# Ask a question
user_input = input('Please enter your question: ')

while user_input != 'exit':
    result = qa.invoke({"query": user_input})

    print("\n\nANSWER:\n")
    print(result["result"])

    print("\n\nSOURCES:\n")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\n--- DOC {i} ---")
        print(f"Source: {doc.metadata.get('source')} | page: {doc.metadata.get('page')}")
        print(doc.page_content[:800]) 

    user_input = input('Please enter your question: ')
