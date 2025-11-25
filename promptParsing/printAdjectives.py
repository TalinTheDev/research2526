import spacy
from spacy.lang.en.examples import sentences 

nlp = spacy.load("en_core_web_sm")
doc = nlp("10 year old boy in red winter clothing")
print(doc.text)

adjectives = []
for token in doc:
    if(spacy.explain(token.pos_) == "adjective"):
        adjectives.append(token.text)

for adj in adjectives:
    print(adj)