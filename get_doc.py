import os
import re
import json
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.chat_models.tongyi import ChatTongyi  # Qwen (DashScope)


CLINICAL_FOLDER_PATH = "Clinical Files"
CLAIMS_PATH = "Flublok_Claims.json"
CLAIM_MATCH_OUTPUT = "claim_doc_search.json"
CLAIM_RAG_OUTPUT = "claim_rag_answers.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Qwen config (DashScope). Set env: DASHSCOPE_API_KEY
QWEN_MODEL_DEFAULT = os.getenv("QWEN_MODEL", "qwen3-72b-instruct")  # adjust if you prefer another Qwen-3 variant
QWEN_TEMPERATURE_DEFAULT = float(os.getenv("QWEN_TEMPERATURE", "0.2"))
QWEN_MAX_TOKENS_DEFAULT = int(os.getenv("QWEN_MAX_NEW_TOKENS", "768"))


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
    docs: List[Document],
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


def build_qwen_llm(
    model: Optional[str] = None,
    *,
    temperature: float = QWEN_TEMPERATURE_DEFAULT,
    max_tokens: int = QWEN_MAX_TOKENS_DEFAULT
):
    model_name = model or QWEN_MODEL_DEFAULT
    llm = ChatTongyi(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm


def _format_ctx_block(docs: List[Document]):

    blocks = []
    for idx, d in enumerate(docs, start=1):
        file = d.metadata.get("file", "UNKNOWN")
        page = d.metadata.get("page", "UNK")
        snippet = d.page_content.strip()
        blocks.append(f"[{idx}] {file} p.{page}\n{snippet}")
    return "\n\n".join(blocks)


def build_qwen_prompt(system_instruction: Optional[str] = None):

    sys_msg = system_instruction or (
        "You are a clinical evidence assistant. Given the user's claim and the provided document context, "
        "decide whether the claim is Supported, Partially supported, Not supported, or Insufficient information. "
        "Keep the answer concise and cite sources using bracket indices like [1], [2] that correspond to the provided context. "
        "If unsure, say so explicitly."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_msg),
            ("user",
             "Claim:\n{claim}\n\n"
             "Context (use for grounding; cite with [#]):\n{context}\n\n"
             "Respond in the following JSON keys:\n"
             '{"verdict":"Supported|Partially supported|Not supported|Insufficient information",'
             '"answer":"short explanation with [#] citations"}'),
        ]
    )
    return prompt


def generate_answer_qwen(
    claim: str,
    ctx_docs: List[Document],
    *,
    llm: Optional[ChatTongyi] = None,
    system_instruction: Optional[str] = None
):

    if llm is None:
        llm = build_qwen_llm()

    prompt = build_qwen_prompt(system_instruction)
    chain = prompt | llm | StrOutputParser()

    context_block = _format_ctx_block(ctx_docs)
    try:
        out = chain.invoke({"claim": claim, "context": context_block})
    except Exception as e:
        out = json.dumps({
            "verdict": "Insufficient information",
            "answer": f"LLM error: {str(e)}"
        })
    return out


if __name__ == "__main__":
    # 1) Ingest & index
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

    # Load claims
    claims = read_json(CLAIMS_PATH)

    # Retrieve for each claim (for auditing), and generate grounded verdict/answer with Qwen-3.
    searched_similar_quote = []
    rag_answers = []

    # Prepare LLM
    llm = build_qwen_llm()

    for c in claims.get('claims', []):
        claim: str = c.get("claim", "")
        retrieved_docs = faiss_search(vs, claim, k=5)

        match_source = []
        for i in retrieved_docs:
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

        # Grounded generation
        llm_json_str = generate_answer_qwen(
            claim,
            retrieved_docs,
            llm=llm
        )

        parsed = {"verdict": "Insufficient information", "answer": llm_json_str}
        try:
            maybe = json.loads(llm_json_str)
            if isinstance(maybe, dict) and "verdict" in maybe and "answer" in maybe:
                parsed = maybe
        except Exception:
            pass

        rag_answers.append(
            {
                "claim": claim,
                "verdict": parsed.get("verdict"),
                "answer": parsed.get("answer"),
                "citations": [
                    {
                        "document_name": md.get("document_name"),
                        "document_page": md.get("document_page")
                    } for md in match_source
                ]
            }
        )

    save_json({"claims": searched_similar_quote}, CLAIM_MATCH_OUTPUT)
    save_json({"claims": rag_answers}, CLAIM_RAG_OUTPUT)

    print(f"Saved retrieval matches → {CLAIM_MATCH_OUTPUT}")
    print(f"Saved RAG answers → {CLAIM_RAG_OUTPUT}")
