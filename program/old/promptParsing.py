from ollama import chat
from pydantic import BaseModel

class Target(BaseModel):
    object: str
    state: list[str]
    descriptions: list[str]
    associated_objects: list[str]
    colors: list[str]
    size_hint: str
    attributes: list[str]

prompt = input("What should we look for? > ")

SYSTEM_PROMPT = """
You are an information extraction and expansion system for a visual search robot.

Extract structured data from the user's search request and return ONLY a valid JSON object matching the schema below.

Rules:
- Return only the JSON — no explanations, no extra text, no markdown
- If information is missing, use empty lists or null
- Be concise and normalize wording
- Think carefully about what a robot camera might actually see for each field

Field definitions:

  object
    The primary target being searched for.
    Examples: "child", "person", "dog"

  state
    Conditions or situations describing the target's current status.
    Examples: ["lost"], ["injured"], ["trapped"], ["moving"]

  descriptions
    Physical or visual descriptors of the target itself.
    Split into as many atomic clues as possible — one descriptor per entry.
    Examples: ["red jacket", "small", "winter clothing", "dark hair"]

  associated_objects
    Objects that are commonly carried by or found near the target, based on
    the descriptions and context. These may appear in the camera even when
    the target itself is partially hidden or occluded.
    Examples: for a child with a backpack → ["backpack", "bag"]
              for a hiker              → ["backpack", "walking stick", "hat", "boots"]
              for an injured person    → ["stretcher", "first aid kit"]

  colors
    All specific colors mentioned or strongly implied by the descriptions.
    Extract every color word — include clothing colors, object colors, etc.
    Examples: ["red", "blue", "orange"]

  size_hint
    A single word describing the expected visual size of the target in frame
    when the robot is at a useful detection range.
    One of: "tiny", "small", "medium", "large"
    Base this on the type of object and any size descriptors in the request.
    Examples: child → "small", adult → "medium", vehicle → "large"

  attributes
    Abstract or contextual attributes not captured by the above fields.
    Include clothing types, materials, conditions, and any other search cues.
    Examples: ["winter clothing", "outdoor gear", "camouflage", "reflective vest"]

Normalize synonyms:
  - "kid", "boy", "girl", "child", "toddler", "youth" → "child"
  - "human", "person", "figure", "human shape", "adult", "man", "woman" → "person"
  - "pup", "puppy" → "dog"
  - "feline", "kitty" → "cat"

Example input:
  "find a lost child wearing a red winter jacket and carrying a small blue backpack"

Example output:
  {
    "object":             "child",
    "state":              ["lost"],
    "descriptions":       ["red winter jacket", "small", "blue backpack", "winter clothing"],
    "associated_objects": ["backpack", "jacket", "hat", "gloves", "winter gear"],
    "colors":             ["red", "blue"],
    "size_hint":          "small",
    "attributes":         ["winter clothing", "outdoor gear"]
  }

Return only the JSON.
"""

response = chat(
    model="llama3.1",
    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
    format=Target.model_json_schema(),
    stream=False)

target = Target.model_validate_json(response.message.content if response.message.content else "")
print(target)