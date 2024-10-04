from flask import Flask, request, render_template
import os
import re
from collections import defaultdict
from math import log, sqrt

app = Flask(__name__)

# Preprocessing function
def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load documents
def load_documents(folder_path):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                docs[filename] = preprocess(file.read())
    return docs

# Load queries from file
def load_queries(query_file):
    queries = []
    with open(query_file, 'r') as file:
        for line in file:
            queries.append(line.strip())  # Strip newline and spaces
    return queries

# Compute term frequencies and document frequencies for TF-IDF
def compute_statistics(docs):
    doc_count = len(docs)
    term_doc_freq = defaultdict(int)
    term_freq = defaultdict(lambda: defaultdict(int))

    for doc_id, words in docs.items():
        word_set = set(words)
        for word in words:
            term_freq[doc_id][word] += 1
        for word in word_set:
            term_doc_freq[word] += 1

    return term_freq, term_doc_freq, doc_count

# Compute TF-IDF
def compute_tfidf(term_freq, term_doc_freq, doc_count):
    tfidf = defaultdict(lambda: defaultdict(float))
    for doc_id, terms in term_freq.items():
        for term, freq in terms.items():
            tf = freq / sum(terms.values())
            idf = log(doc_count / (term_doc_freq[term] + 1))
            tfidf[doc_id][term] = tf * idf
    return tfidf

# Function to calculate cosine similarity
def cosine_similarity(query_vec, doc_vec):
    dot_product = sum(query_vec.get(term, 0) * doc_vec.get(term, 0) for term in query_vec)
    query_magnitude = sqrt(sum(v ** 2 for v in query_vec.values()))
    doc_magnitude = sqrt(sum(v ** 2 for v in doc_vec.values()))
    return dot_product / (query_magnitude * doc_magnitude + 1e-10)

# Global variables to store precomputed data
docs = {}
tfidf = {}

@app.before_first_request
def startup():
    global docs, tfidf
    folder_path = "Dataset_ IR"  # Update this path
    docs = load_documents(folder_path)
    term_freq, term_doc_freq, doc_count = compute_statistics(docs)
    tfidf = compute_tfidf(term_freq, term_doc_freq, doc_count)

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        top_k = int(request.form.get('top_k', 5))
        
        query_terms = preprocess(query)
        query_tfidf = {term: 1 for term in query_terms}  # Simple TF-IDF for query
        
        scores = {}
        for doc_id, doc_tfidf in tfidf.items():
            scores[doc_id] = cosine_similarity(query_tfidf, doc_tfidf)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_scores[:top_k]
        
        return render_template('results.html', results=top_results, query=query)
    
    return render_template('search.html')

if __name__ == '__main__':
    app.run(debug=True)


 