import os
import re
import json
from typing import List, Dict
from retry import retry
from dotenv import load_dotenv

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

Provide a numeric score between 0 and 10. **Return only the number.**
""",
    input_variables=["question", "answer"],
)


# 3) Utility to pull out the numeric 0–10 score
def extract_numeric_score(text: str) -> float:
    match = re.search(r"\b(10(?:\.0)?|[0-9](?:\.[0-9])?)\b", text)
    return float(match.group()) if match else -1.0

# 4) Single‐model scoring with retry
@retry(tries=3, delay=1)
def score_response(question: str, answer: str, model_key: str) -> float:
    llm = LLMS[model_key]
    # Chain the prompt + LLM
    chain = prompt_template | llm
    # Invoke the chain — pass a dict matching your input_variables
    result = chain.invoke({"question": question, "answer": answer})
    # result is a ChatGeneration or similar; grab the content
    text = result.content if hasattr(result, "content") else result
    return extract_numeric_score(text.strip())

# 5) Orchestrator over a candidate’s `responses.json`
def score_qa_pairs(qa_pairs: List[Dict[str,str]]) -> List[Dict]:
    results = []
    for pair in qa_pairs:
        q = pair.get("question","").strip()
        a = pair.get("answer","").strip()
        if not q or not a:
            continue
        scores: Dict[str,float] = {}
        valid: List[float] = []
        for key, field in mapping.items():
            print(f"[{key}] scoring…")
            s = score_response(q, a, key)
            scores[field] = s
            if s >= 0:
                valid.append(s)
        avg = round(sum(valid)/len(valid),2) if valid else -1.0
        results.append({
            "question": q,
            "answer": a,
            **scores,
            "average_score": avg
        })
    return results

# Example usage without any JSON file:
if __name__ == "__main__":
    # define your Q&A in code (or load them however you like)
    sample_qa = [
        {"question":"What is an operating system?","answer":"An operating system (OS) is system software that manages computer hardware and software resources and provides common services for computer programs. It handles tasks such as process management, memory allocation, device control, file management, security, and user interfaces, enabling applications to run efficiently and reliably."},
        {"question":"Explain deadlock.","answer":"Deadlock is a concurrency issue in multitasking environments where two or more processes become blocked waiting indefinitely for resources held by each other, preventing progress. This situation arises when mutual exclusion, hold and wait, no preemption, and circular wait conditions coexist. Techniques like resource ordering, avoidance algorithms, and detection with recovery help mitigate deadlocks effectively."},
    ]
    scored = score_qa_pairs(sample_qa)
    print(json.dumps(scored, indent=2))
