import os
import re
import json

from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import DistanceStrategy


CLINICAL_FOLDER_PATH = "Clinical Files"
CLAIMS_PATH = "Flublok_Claims.json"
CLAIM_OUTPUT_DIR = "clamin_doc_search.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def pdf_reader(path: str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    pages = [d.page_content for d in docs]
    pages = [i.replace("\n", " ") for i in pages]
    
    return pages


def chunk_doc(
    pages: List[str],
    *,
    file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[". ", ""],
    )

    page_texts: List[str] = pages
    metadatas = [{"file": file, "page": i + 1} for i in range(len(page_texts))]
    docs = splitter.create_documents(page_texts, metadatas=metadatas)
    for d in docs:
        d.page_content = d.page_content.lstrip(". ").strip()
    return docs


def build_faiss(
    docs: List,
    *,
    faiss_dir: str = "faiss_file"
):
    vs = FAISS.from_documents(docs, EMBEDDING, distance_strategy=DistanceStrategy.COSINE)
    if faiss_dir:
        vs.save_local(faiss_dir)
    return vs, EMBEDDING


def load_faiss(faiss_dir: str):
    
    vs = FAISS.load_local(faiss_dir, EMBEDDING, allow_dangerous_deserialization=True)
    return vs, EMBEDDING


def faiss_search(vs: FAISS, query: str, *, k: int = 5):
    results = vs.similarity_search_with_score(query, k=k)
    # TODO: Filter by similarity score
    # return results
    return vs.similarity_search(query, k=k)

def read_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_json(data: dict, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def get_table(text: str):
    pattern = r"\b(?:Table|Fig)\.?\s*\d+[A-Za-z]?\b"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return matches


if __name__ == "__main__":

    all_docs = []
    for fname in os.listdir(f"{CLINICAL_FOLDER_PATH}"):
        if not fname.lower().endswith(".pdf"):
            continue
        path = f"{CLINICAL_FOLDER_PATH}/{fname}"

        pages_or_text = pdf_reader(path)

        all_docs.extend(
            chunk_doc(
                pages_or_text,
                file=fname,
                chunk_size=1000,
                chunk_overlap=200
            )
        )

    vs, emb = build_faiss(all_docs)
    claims = read_json(CLAIMS_PATH)

    searched_similar_quote = []
    for c in claims['claims']:
        claim: str = c["claim"]
        r = faiss_search(vs, claim, k=5)

        match_source = []
        for i in r:
            output_struct = {
                "document_name": i.metadata.get('file'),
                "document_page": i.metadata.get('page'),
                "matching_text": i.page_content,
            }

            labels = get_table(i.page_content)
            if labels:
                output_struct["matching_table_fig"] = labels

            match_source.append(output_struct)

        searched_similar_quote.append(
            {
                "claim": claim,
                "match_source": match_source
            }
        )

    save_json({"claims": searched_similar_quote}, CLAIM_OUTPUT_DIR)