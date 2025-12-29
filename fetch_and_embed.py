import pandas as pd
import wikipediaapi

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize, word_tokenize

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
wiki = wikipediaapi.Wikipedia(user_agent='WikiRag', language='en')
TOPICS = ['solar system', 'the sun', 'mercury (planet)', 'venus', 'earth', 'mars', 'asteroid belt', 'jupiter', 'saturn',
          'uranus', 'neptune', 'the moon', 'formation and evolution of the solar system', 'solar eclipse',
          'lunar eclipse', 'coronal mass ejection', 'solar wind', 'solar activity and climate', 'life on mars',
          'galilean moons', 'ring rings', 'rings of jupiter', 'rings of saturn', 'rings of uranus', 'rings of neptune',
          'pluto', 'halleys comet', 'dwarf planets' ,'comet', 'kuiper belt', 'heliosphere', 'oort cloud']

def chunk_article(article, max_tokens=150, overlap=10):
    sentences = sent_tokenize(article.text)
    article_chunks, current_chunk, tokens = [], '', 0
    for sentence in sentences:
        sentence_length = len(word_tokenize(sentence))
        if sentence_length + tokens < max_tokens:
            current_chunk += ' ' + sentence
            tokens += sentence_length
        else:
            article_chunks.append(current_chunk)
            overlap_string = ' '.join(word_tokenize(current_chunk)[-overlap:])
            current_chunk = overlap_string + sentence
            tokens = len(word_tokenize(current_chunk))

    # Last chunk
    if current_chunk:
        article_chunks.append(current_chunk)
    return article_chunks

def chunk_article_into_sentences(article):
    sentences = sent_tokenize(article.text)
    article_chunks = []
    for sentence in sentences:
        article_chunks.append(sentence)

    return article_chunks

df = pd.DataFrame(columns=['text', 'embedding', 'title', 'url'])
for i, topic in tqdm(enumerate(TOPICS), total=len(TOPICS)):
    page = wiki.page(topic)
    if page.exists():
        chunked_article = chunk_article(page)
        embeddings = model.encode(chunked_article)
        url = f'https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}.'
        new_rows = pd.DataFrame({
            'text': chunked_article,
            'embedding': [emb.tolist() for emb in embeddings],
            'title': [page.title] * len(chunked_article),
            'url': url * len(chunked_article),
        })
        df = pd.concat([df, new_rows], ignore_index=True)

df.to_csv('data/overlapped_embeddings_150_tokens_10_overlap.csv', index=True)

