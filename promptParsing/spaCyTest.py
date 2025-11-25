import spacy
from spacy.lang.en.examples import sentences 

nlp = spacy.load("en_core_web_sm")
doc = nlp("10 year old boy in red winter clothing")
print(doc.text)
for token in doc:
    print(token.text, " | ", spacy.explain(token.pos_))