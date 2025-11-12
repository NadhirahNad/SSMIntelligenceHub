from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from retriever import retrieve_context
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load small language model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

@app.get("/")
def root():
    return {"message": "SSM Intelligence Hub Backend is running!"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("question")
    act_name = data.get("act_name")

    # Retrieve relevant context
    context = retrieve_context(query, act_name)

    # Generate answer
    input_text = f"Answer based on law: {context} \nQuestion: {query}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=200, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer}
