# 📌 Importation des bibliothèques nécessaires
import fitz
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import itertools
import openpyxl
import multiprocessing
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# Fonction pour extraire le texte d'un fichier PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for page in doc])
    return text

# Fonction de prétraitement du texte
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Fonction pour calculer les scores TF-IDF et sélectionner les mots les plus pertinents
def compute_top_tfidf_words(tokens1, tokens2, top_n=500):  # Réduit le nombre de mots à 500
    doc1, doc2 = " ".join(tokens1), " ".join(tokens2)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores1 = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    tfidf_scores2 = dict(zip(feature_names, tfidf_matrix.toarray()[1]))
    return sorted(tfidf_scores1, key=tfidf_scores1.get, reverse=True)[:top_n], sorted(tfidf_scores2, key=tfidf_scores2.get, reverse=True)[:top_n]

# Fonction pour calculer la fréquence des tokens et leurs probabilités
def compute_token_probabilities(tokens, total_tokens):
    token_counts = Counter(tokens)
    return {word: count / total_tokens for word, count in token_counts.items()}

# Correction du problème de pickling : Utilisation d'une fonction au lieu de `lambda`
def default_dict():
    return defaultdict(int)

# Construction de la matrice de co-occurrence
def build_cooccurrence_matrix(tokens, window=10):
    cooccurrence_matrix = defaultdict(default_dict)  # ✅ Correction ici
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window, len(tokens))):
            token1, token2 = tokens[i], tokens[j]
            cooccurrence_matrix[token1][token2] += 1
            cooccurrence_matrix[token2][token1] += 1
    return cooccurrence_matrix

# Fonction pour calculer la probabilité conjointe
def compute_joint_probability(token1, token2, cooccurrence_matrix, total_windows):
    count_joint = cooccurrence_matrix[token1].get(token2, 0)
    return count_joint / total_windows if count_joint > 0 else 0

# Optimisation du calcul du MI
def compute_mi(token1, token2, prob_t1, prob_t2, joint_prob):
    EPSILON = 1e-10
    prob_t1, prob_t2, joint_prob = max(prob_t1, EPSILON), max(prob_t2, EPSILON), max(joint_prob, EPSILON)
    mi = joint_prob * np.log2(joint_prob / (prob_t1 * prob_t2))

    if abs(mi) > 1e-5:  # Afficher uniquement les MI significatifs
        print(f"🔍 MI significatif: {token1}-{token2} -> MI={mi}")

    return mi

# Fonction pour calculer l'entropie conjointe
def compute_joint_entropy(joint_prob):
    return -joint_prob * np.log2(joint_prob) - (1 - joint_prob) * np.log2(1 - joint_prob) if 0 < joint_prob < 1 else 0

def compute_pwi(mi, h):
    return mi - h

# Calcul du PWI pour un chunk
def compute_pwi_for_pairs(pair_list, probabilities, cooccurrence_matrix, total_windows):
    results = []
    for idx, (t1, t2) in enumerate(pair_list):
        joint_prob = compute_joint_probability(t1, t2, cooccurrence_matrix, total_windows)
        mi = compute_mi(t1, t2, probabilities.get(t1, 0), probabilities.get(t2, 0), joint_prob)
        h = compute_joint_entropy(joint_prob)
        pwi = compute_pwi(mi, h)
        results.append((t1, t2, pwi))
    return results

def process_chunk(args):
    chunk, probabilities, cooccurrence_matrix, total_windows = args
    return compute_pwi_for_pairs(chunk, probabilities, cooccurrence_matrix, total_windows)

# Partie principale
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    pdf1_path, pdf2_path = "Spectra.pdf", "Spectra2.pdf"
    text1, text2 = extract_text_from_pdf(pdf1_path), extract_text_from_pdf(pdf2_path)
    tokens1, tokens2 = preprocess_text(text1), preprocess_text(text2)
    top_tokens1, top_tokens2 = compute_top_tfidf_words(tokens1, tokens2)

    print(f"🔍 Top tokens Article 1: {top_tokens1[:20]}")
    print(f"🔍 Top tokens Article 2: {top_tokens2[:20]}")

    print(" Construction de la matrice de co-occurrence avec `window=10`...")
    cooccurrence_matrix = build_cooccurrence_matrix(tokens1 + tokens2, window=10)
    total_windows = len(tokens1) + len(tokens2)
    print("✅ Matrice construite.")

    probabilities = compute_token_probabilities(tokens1 + tokens2, total_windows)
    token_pairs = list(itertools.product(top_tokens1, top_tokens2))

    num_cores = min(multiprocessing.cpu_count(), 8)
    chunk_size = max(10, len(token_pairs) // num_cores)  # Augmenté pour réduire le nombre de tâches
    chunks = [token_pairs[i:i + chunk_size] for i in range(0, len(token_pairs), chunk_size)]
    args_list = [(chunk, probabilities, cooccurrence_matrix, total_windows) for chunk in chunks]

    print("Lancement du multiprocessing...")
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(process_chunk, args_list)

    pd.DataFrame([item for sublist in results for item in sublist], columns=["Token1", "Token2", "PWI"]).to_excel("pwi_parallel_optimized.xlsx", index=False)
    print("Les résultats ont été enregistrés.")
