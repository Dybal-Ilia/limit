from transformers import pipeline

class QueryPlanningAgent:
    def __init__(self, model_name="sshleifer/tiny-gpt2", max_new_tokens: int = 64):
        self.llm = pipeline("text-generation", model=model_name)
        self.max_new_tokens = max_new_tokens

    def decompose_query(self, instruction):
        prompt = f"Decompose this complex search query into sub-queries:\n{instruction}"
        response = self.llm(prompt, max_new_tokens=self.max_new_tokens)[0]["generated_text"]
        # Heuristic split by lines and bullets
        lines = [q.strip() for q in response.split("\n") if q.strip()]
        parts = []
        for ln in lines:
            ln = ln.strip("-• ")
            if ln:
                parts.append(ln)
        # Fallback to the original instruction if decomposition failed
        return parts or [instruction]
