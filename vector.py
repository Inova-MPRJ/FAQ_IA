import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_milvus import Milvus

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

URI = "tcp://localhost:19530"  

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
)

collection_name = "faq_collection2"


def indexer(files, folder, method='document', chunk_size=2000, chunk_overlap=200):
    docs = []
    for file in files:
        loader = PyPDFLoader(os.path.join(folder, file))
        doc = loader.load()

        if method == 'document':
            complete_text = " ".join([page.page_content for page in doc])
            docs.append(Document(page_content = complete_text))

        elif method == 'chunk':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            splits = text_splitter.split_documents(doc)
            docs.extend(splits)

        elif method == 'page':
            docs.extend(doc)

        elif method == 'paragraph':
            for page in doc:
                for paragraph in page.paragraphs:
                    docs.append(Document(page_content = paragraph))
    return docs



if __name__ == "__main__":
    folder = r"/home/lcolimerio/workspace/FAQ_IA/assets"
    docs = indexer(os.listdir(folder), folder, method='chunk')

    vector_store_saved = Milvus.from_documents(
        docs,
        embeddings,
        collection_name=collection_name,
        connection_args={"uri": URI},
    )

    print(f"{len(docs)} documentos foram vetorados e salvos no Milvus.")