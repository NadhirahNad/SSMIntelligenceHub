from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import re
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

print("Loaded HF token:", HF_TOKEN[:10] + "...")


# === Neo4j Connection ===
uri = "neo4j://127.0.0.1:7687"
username = "neo4j"
password = "ssm_database"
driver = GraphDatabase.driver(uri, auth=(username, password))

# === Models ===
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

# Load LLaMA-3 (choose 8B or 70B depending on your machine/GPU)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # puts model on GPU if available
    torch_dtype=torch.float16,
)

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.2,  # keep it deterministic
    top_p=0.9
)

# === Extract Section Number from Query ===
def extract_section_number(query):
    match = re.search(r'section\s+(\d+[A-Za-z\-]*)', query, re.IGNORECASE)
    return match.group(1) if match else None

# === Extract Act Name from Query ===
def extract_act_name(query):
    query = query.lower()
    if "roc 2016" in query or "companies act" in query or "companies 2016" in query:
        return "ROC 2016"
    elif "rob 1956" in query:
        return "ROB 1956"
    return None

# === Retrieval Function Based on Neo4j Structure (Hierarchical) ===
def retrieve_content_from_neo4j(query_text, top_k=3):
    section_number = extract_section_number(query_text)
    act_name = extract_act_name(query_text)

    with driver.session() as session:
        query = """
        MATCH (a:Act)-[:HAS_PART|HAS_SECTION*]->(s:Section)-[:HAS_CONTENT]->(c:Content)
        OPTIONAL MATCH (s)<-[:HAS_SECTION]-(p:Part)
        WHERE ($act_name IS NULL OR a.name = $act_name)
        RETURN a.name AS act, s.name AS section, p.name AS part, c.text AS content
        """
        results = session.run(query, {"act_name": act_name}).data()

    meta_info = []
    texts_for_embedding = []

    for record in results:
        act = record["act"] or "Unknown Act"
        section = record["section"] or "Unknown Section"
        part = record["part"] or "No Part"
        content = record["content"] or ""

        combined_text = f"{act} {part} {section} {content}"
        texts_for_embedding.append(combined_text)
        meta_info.append((act, section, part, content))

    if section_number:
        exact_matches = []
        for act, section, part, content in meta_info:
            cleaned_section = re.sub(r'[^0-9A-Za-z\-]', '', section).lower()
            if section_number.lower() == cleaned_section:
                exact_matches.append({
                    "act": act,
                    "section": section,
                    "part": part,
                    "content": content,
                    "similarity": 1.0,
                    "is_deleted": bool(re.search(r"(deleted|repealed|revoked)", content, re.IGNORECASE))
                })
        if exact_matches:
            return exact_matches[:top_k]

    if not texts_for_embedding:
        return []

    query_embedding = embedding_model.encode([query_text])
    candidate_embeddings = embedding_model.encode(texts_for_embedding)
    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

    scored = []
    for i, (act, section, part, content) in enumerate(meta_info):
        score = similarities[i]
        scored.append({
            "act": act,
            "section": section,
            "part": part,
            "content": content,
            "similarity": float(score),
            "is_deleted": bool(re.search(r"(deleted|repealed|revoked)", content, re.IGNORECASE))
        })

    top = sorted(scored, key=lambda x: x["similarity"], reverse=True)[:top_k]
    return top

# === Main Bot Function ===
def ask_ssm_bot(query, top_k=3):
    top_results = retrieve_content_from_neo4j(query, top_k)

    if not top_results:
        act = extract_act_name(query)
        section = extract_section_number(query)

        with driver.session() as session:
            acts = session.run("MATCH (a:Act) RETURN DISTINCT a.name").value()
            sections = []
            if act:
                sections = session.run(
                    "MATCH (a:Act {name: $act})-[:HAS_PART|HAS_SECTION*]->(s:Section) RETURN s.name",
                    {"act": act}
                ).value()

        if act in acts and section:
            return f"!!! Section {section} not found under {act}. Available sections: {', '.join(sections[:5])}...", "0.00%", []

        return "!!! No relevant legal content found for the specified Act and section.", "0.00%", []

    first_result = top_results[0]
    if first_result['is_deleted']:
        return (
            f"🗑️ Section {first_result['section']} of {first_result['act']} has been deleted/repealed.\n\n{first_result['content']}",
            "100.00%",
            top_results
        )

    context_blocks = [
        f"[Act: {item['act']} | Part: {item['part']} | Section: {item['section']}]\n{item['content']}"
        for item in top_results
    ]
    context = "\n".join(dict.fromkeys("\n".join(context_blocks).split("\n")))

    prompt = (
        "You are a legal document assistant. Answer **only** using the retrieved legal content provided below. "
        "Do not use any external knowledge.\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"User Question:\n{query}\n\n"
        "Answer:"
    )

    result = qa_pipeline(prompt)[0]['generated_text']
    avg_score = sum([item["similarity"] for item in top_results]) / len(top_results)
    confidence = f"{(avg_score * 100):.2f}%"

    return result.strip(), confidence, top_results
    