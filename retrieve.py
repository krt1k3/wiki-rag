import ast
import pandas as pd
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
language_model = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = 'which planet has the volcano olympus mons'
query_emb = embedding_model.encode(query)
df = pd.read_csv('data/overlapped_embeddings.csv')
df['embedding'] = df['embedding'].apply(lambda s: np.array(ast.literal_eval(s)))
df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_emb, x))
df.sort_values('similarity', ascending=False, inplace=True)

context = df.iloc[0]


content = f'''
Answer the Query using ONLY the provided context. 
Generate response strictly for the query. 
DO NOT add other details\n\n
Query: {query}\n
Context:\n{context.text}
'''
messages = [
    {
        'role': "user",
        'content': content
    },
]

response = ollama.chat(model=language_model, messages=messages)
confidence_score = f'{(context.similarity * 100):.2f}'
print(f'{response.message.content}\n\nConfidence score: {confidence_score}')
