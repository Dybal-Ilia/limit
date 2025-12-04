from transformers import pipeline
import json
import re

class QueryPlanningAgent:
    def __init__(self, model_name: str = "google/flan-t5-small", max_new_tokens: int = 64, task: str | None = None):
        """
        Lightweight agent for query decomposition.
        Defaults to an instruction-tuned seq2seq model for cleaner, structured outputs.

        Parameters:
        - model_name: HF model id. Prefer instruction-tuned models (e.g., flan-t5-*)
        - max_new_tokens: generation cap
        - task: optional transformers pipeline task override
        """
        # Prefer text2text-generation for instruction-tuned models; fallback to text-generation
        self.task = task or ("text2text-generation" if "flan" in model_name or "t5" in model_name else "text-generation")
        self.llm = pipeline(self.task, model=model_name)
        self.max_new_tokens = max_new_tokens

    def _build_prompt(self, instruction: str) -> str:
        # Ask for strict JSON list for robust parsing
        return (
            "You are a query planning assistant. "
            "Given a complex user query, produce ONLY a JSON array of short, clean sub-queries. "
            "Each sub-query must be under 12 words, contain only letters, numbers, spaces and punctuation, "
            "and must be meaningful English phrases relevant to the user query. "
            "Do not include explanations or any text outside the JSON array.\n\n"
            f"User query: {instruction}\n\n"
            "Output format example: [\"who likes pizza\", \"famous pizza fans\", \"pizza popularity history\"]"
        )

    def decompose_query(self, instruction: str):
        """
        Return a list of sub-queries. Uses structured prompting and strict post-processing
        to avoid gibberish. Always returns at least one clean sub-query.
        """
        prompt = self._build_prompt(instruction)
        try:
            gen = self.llm(prompt, max_new_tokens=max(32, min(self.max_new_tokens, 128)))
            text = gen[0].get("generated_text") or gen[0].get("translation_text") or gen[0].get("summary_text") or ""
        except Exception:
            text = ""

        # Helper: validate and clean a candidate sub-query
        def _clean(s: str) -> str | None:
            s = s.strip()
            if not s:
                return None
            # Remove repeated non-letter sequences and excessive length
            # Keep only reasonable characters
            s = re.sub(r"[^\w\s\-,'?()]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            # Reject if too long or contains long nonsensical token runs
            if len(s.split()) > 12:
                s = " ".join(s.split()[:12])
            # Basic heuristic: must have at least 2 alphabetic words
            alpha_words = [w for w in s.split() if re.search(r"[A-Za-z]", w)]
            if len(alpha_words) < 2:
                return None
            return s

        parts: list[str] = []
        # Try JSON parsing first
        if text:
            try:
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    candidate = text[start : end + 1]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        for p in parsed:
                            if isinstance(p, str):
                                cs = _clean(p)
                                if cs:
                                    parts.append(cs)
            except Exception:
                parts = []

        # Fallback: heuristic split on lines/bullets if JSON failed
        if not parts and text:
            for ln in text.split("\n"):
                s = ln.strip().strip("-• ")
                if not s:
                    continue
                # ignore boilerplate and prompt echoes
                low = s.lower()
                if low.startswith("you are a") or low.startswith("output format") or low.startswith("user query"):
                    continue
                cs = _clean(s)
                if cs:
                    parts.append(cs)

        # Deduplicate while preserving order
        seen = set()
        clean_parts = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                clean_parts.append(p)

        # Final fallback: derive simple sub-queries from the instruction
        if not clean_parts:
            base = instruction.strip()
            base = re.sub(r"\s+", " ", base)
            if base:
                # produce 1-2 variants
                clean_parts = [base]
                if not base.lower().startswith("who"):
                    clean_parts.append(f"Who {base}")
            else:
                clean_parts = ["Find relevant information"]

        return clean_parts
