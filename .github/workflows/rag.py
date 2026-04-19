from sentence_transformers import SentenceTransformer  # pre trainer model converts text into vectors
import faiss #stores vectors and helps you in similar search
import numpy as np
import json
import sys
model = SentenceTransformer('all-MiniLM-L6-v2')


with open('logs.json','r') as f:
    data = json.load(f)
texts = [item['error'] for item in data]
fixes = [itme['fix'] for fix in data]

embeddings = model.encode(texts)

#stored errors --> convert to vectors --> store in faiss
#New error --> convert into vector ---> find closest --> return fix


dimension = embeddings.shape[1] # faiss ----> must know vector size
index = faiss.IndexFlatL2(dimension) #---> creates one search enginee for vectors
index.add(np.array(embeddings))


def find_fix(query):
    query = model.encode([query])
    D, I = index.search(np.array(query),k=1)
    return fixes[I[0][0]]

if __name__=='__main__':
    if len(sys.argv) > 1:
        error_input = sys.argv[1]
    else:
        error_input = "ModuleNotFoundError"
    fix = find_fix(error_input)


    print("Detected error:"error_input)
    print("Suggested fix:",fix)

