import ast
import pandas as pd
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
language_model = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = 'which moon is the largest'
query_emb = embedding_model.encode(query)
df = pd.read_csv('data/overlapped_embeddings.csv')
df['embedding'] = df['embedding'].apply(lambda s: np.array(ast.literal_eval(s)))
df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_emb, x))
df.sort_values('similarity', ascending=False, inplace=True)
df = df[df['similarity'] > 0.5]

if len(df.index) > 3:
    df = df[df.index < 4]

similarity = np.average(np.array(df['similarity']))
context = np.array(df['text'])


content = f'''
Answer the Query USING ONLY the provided Context.
If the answer is no clear in the Context respond:
"Answer not found"\n
Query: {query}\n
Context:\n{context}
'''
messages = [
    {
        'role': "user",
        'content': content
    },
]

response = ollama.chat(model=language_model, messages=messages)
confidence_score = f'{(similarity * 100):.2f}'
print(f'{response.message.content}\n\nConfidence score: {confidence_score}\n\n{context}')
