from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re

def grade_answer(llm, query, ground_truth, generated_answer):
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic evaluator. Your job is to grade a student AI assistant's generated answer against a definitive Ground Truth answer key.
        
Score the answer strictly on a scale of 1 to 5:
- 5: Excellent. Completely accurate, captures all vital factual points of the ground truth.
- 4: Good. Accurate, but missed a minor detail or is slightly vague.
- 3: Mediocre. Marginally accurate but misses significant factual elements.
- 2: Poor. Mostly inaccurate or contains hallucinated details not supported by the ground truth.
- 1: Irrelevant. The answer completely misses the mark or says it doesn't know.

Output your response strictly in the following JSON format:
{{
    "score": <int_score_from_1_to_5>,
    "reasoning": "<brief_one_sentence_explanation>"
}}"""),
        ("human", f"Question: {query}\n\nGround Truth: {ground_truth}\n\nGenerated Answer: {generated_answer}")
    ])
    
    # Invoke the LLM to grade it
    chain = grader_prompt | llm
    response = chain.invoke({})
    
    try:
        # Simple extraction helper if JSON parsing acts up
        data = json.loads(response.content)
        return data["score"], data["reasoning"]
    except Exception:
        # Fallback if LLM forgets JSON markdown formatting block wraps
        match = re.search(r'"score":\s*(\d)', response.content)
        score = int(match.group(1)) if match else 3
        return score, "Evaluated."