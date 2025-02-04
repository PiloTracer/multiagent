# ./app/ingest_docs.py

import os
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def load_word_documents(directory: str):
    texts = []
    for fname in os.listdir(directory):
        if fname.endswith('.docx'):
            full_path = os.path.join(directory, fname)
            doc_text = docx2txt.process(full_path)
            texts.append(doc_text)
    return texts

def create_vector_store(texts, openai_api_key: str):
    # Split each doc into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for txt in texts:
        chunks = splitter.split_text(txt)
        for c in chunks:
            all_chunks.append(Document(page_content=c))
    
    # Build FAISS from documents
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vs = FAISS.from_documents(all_chunks, embeddings)
    return vs

if __name__ == "__main__":
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set in environment!")

    docs_dir = "./custom_docs"
    doc_texts = load_word_documents(docs_dir)
    if not doc_texts:
        print(f"No .docx files found in '{docs_dir}'. Skipping index creation.")
    else:
        vs = create_vector_store(doc_texts, openai_key)
        vs.save_local("./faiss_store")
        print("Vector store created and saved to ./faiss_store")
