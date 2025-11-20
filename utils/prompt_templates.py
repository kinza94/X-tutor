# utils/prompt_templates.py
# Clean, single-source prompt templates used by app.py
# Make sure this file is saved as utils/prompt_templates.py

EXPLAINER_PROMPT = """
Subject: {subject}
Student question: {question}
Level: {level}

INSTRUCTIONS TO MODEL (READ CAREFULLY):
- Return EXACTLY ONE valid JSON object and NOTHING else.
- JSON keys required:
  - answer: short final answer (string, 1-2 lines)
  - steps: array of strings (each element is one intermediate step; include arithmetic)
  - key_idea: short one-line statement of the core idea (string)
  - hints: array of 1-2 short hints (array of strings)
  - difficulty: string (Easy / Medium / Hard)
  - assumptions: array (if ambiguous, list assumptions)

ADDITIONAL GUIDELINES:
- Do not include any extra text, markdown, or explanation outside the JSON object.
- When possible, produce 3–6 clear numbered steps in the "steps" array.
- Use the subject and level to decide notation, units, and method.
- If you cannot answer (safety/moderation), return a JSON object with an appropriate 'answer' string explaining the restriction and keep steps empty.

Now produce the JSON object for the question above.
"""

PRACTICE_PROMPT = """
Create {n} practice questions based on: {question_or_keyidea}
Level: {level}
Subject: {subject}

INSTRUCTIONS:
- Return EXACTLY ONE JSON object with keys:
  - problems: an array of objects {id:int, prompt:string, type: "MCQ" or "SA", difficulty:string}
  - answers: object mapping id -> answer (string)
- Do NOT output any text outside the JSON object.
"""
