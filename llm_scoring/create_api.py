import os
import re
import json
from typing import List, Dict
from retry import retry
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate

# 1) Load your OpenRouter key from .env
load_dotenv()

# 1) Instantiate your four LLMs
LLMS = {
    "gpt-3.5": ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=1.1,
    ),
    "gemini": ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=1.2,
    ),
    "openchat": ChatTogether(
        model="meta-llama/Llama-Vision-Free",  # free-tier
        temperature=1.3,
    ),
    "meta-llama": ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0.9,
    ),
}

# 2) Map to output fields
mapping = {
    "gpt-3.5":    "gpt_3_5_score",
    "gemini":     "gemini_score",
    "openchat":   "openchat_score",
    "meta-llama": "meta-llama_score",
}

# Build the reusable PromptTemplate once at module‐scope
prompt_template = PromptTemplate(
    template="""
You are an expert technical interviewer. Evaluate the candidate response below.

Question: {question}
Answer: {answer}

Provide a numeric score between 1 and 10. **Return only the number.**
""",
    input_variables=["question", "answer"],
)


# 3) Utility to pull out the numeric 0–10 score
def extract_numeric_score(text: str) -> float:
    match = re.search(r"\b(10(?:\.0)?|[0-9](?:\.[0-9])?)\b", text)
    return float(match.group()) if match else 0

# 4) Single‐model scoring with retry
@retry(tries=3, delay=1)
@retry(tries=3, delay=1)
def score_response(question: str, answer: str, model_key: str) -> float:
    llm = LLMS[model_key]
    chain = prompt_template | llm
    try:
        # Invoke the chain — pass a dict matching your input_variables
        result = chain.invoke({"question": question, "answer": answer})
        # Extract the text
        text = result.content if hasattr(result, "content") else result
        return extract_numeric_score(text.strip())
    except Exception as e:
        # Log what went wrong
        print(f"[{model_key} error] {e}")
        # Return 0 if scoring failed
        return 0.0


# 5) Orchestrator over a candidate’s `responses.json`
def score_qa_pairs(qa_pairs: List[Dict[str, str]]) -> List[Dict]:
    results = []
    for pair in qa_pairs:
        q = pair.get("question", "").strip()
        a = pair.get("answer", "").strip()
        if not q or not a:
            continue
        scores = {}
        valid = []
        for key, field in mapping.items():
            s = score_response(q, a, key)
            scores[field] = s
            if s >= 0:
                valid.append(s)
        average = round(sum(valid) / len(valid), 2) if valid else -1.0
        results.append({
            "question": q,
            "answer": a,
            **scores,
            "average_score": average
        })
    return results

# 6) FastAPI setup
app = FastAPI(title="QA Scoring API")

class QAPair(BaseModel):
    question: str
    answer: str

class ScoreRequest(BaseModel):
    qa_pairs: List[QAPair]

@app.post("/score", summary="Score a list of QA pairs")
def score_endpoint(req: ScoreRequest):
    if not req.qa_pairs:
        raise HTTPException(status_code=400, detail="No QA pairs provided")
    # Convert Pydantic QAPair to plain dicts
    qa_list = [qp.dict() for qp in req.qa_pairs]
    results = score_qa_pairs(qa_list)
    return {"results": results}