from ollama import chat
from pydantic import BaseModel

class Target(BaseModel):
    object: str
    state: list[str]
    descriptions: list[str]

prompt = input("What should we look for? > ")

SYSTEM_PROMPT = """
You are an information extraction system.

Extract structured data from the user's request and return ONLY a valid JSON object that matches the provided schema.

Rules:
- Do not include any explanations or extra text
- Do not include anything outside the JSON
- If information is missing, return empty lists or null values
- Be concise and normalize wording

Field definitions:
- object: the primary target being searched for (e.g. "child", "person", "dog")
- state: conditions or situations (e.g. "lost", "injured", "trapped")
- descriptions: physical or visual descriptors (e.g. "red jacket", "small", "human shape"); these should be split into as many possible descriptions about the object/target (without verbs) as possible in the output array in order to maximize possible search clues

Normalize synonyms:
- "kid", "boy", "girl" → "child"
- "human", "person", "figure", "human shape" → "person"

Return only the JSON.
"""

response = chat(
    model="llama3.1",
    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
    format=Target.model_json_schema(),
    stream=False)

target = Target.model_validate_json(response.message.content if response.message.content else "")
print(target)