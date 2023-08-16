import os
from uu import encode

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == "__main__":
    print("hi")
    pdf_path = "C://Users/saulp/PycharmProjects/vectorstore-in-memory/AML4_2015_849.pdf"

    # loader = PyPDFLoader(file_path=pdf_path)
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    # doc_chunked = text_splitter.split_documents(documents=documents)
    #
    embeddings = OpenAIEmbeddings()
    # vectorstore = FAISS.from_documents(doc_chunked, embeddings)
    # vectorstore.save_local("faiss_index_aml4")

    new_vectorstore = FAISS.load_local("faiss_index_aml4", embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())
    # res = qa.run("Give me the gist of AML4 in 3 sentences")
    # res = qa.run("list all requirements in terms of name screening with the article reference and ID")
    # res = qa.run("what are the requirements regarding the identification of beneficiary owner and list them with the articl reference and ID number")
    res = qa.run("what is a PEP")
    print(res)