import fitz  # PyMuPDF pour extraire le texte des PDF
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd
import itertools
import openpyxl
import multiprocessing
from tqdm import tqdm

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for page in doc])
    return text

# Fonction pour nettoyer et tokeniser le texte
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Fonction pour calculer la fréquence des tokens et leurs probabilités
def compute_token_probabilities(tokens, total_tokens):
    token_counts = Counter(tokens)
    return {word: count / total_tokens for word, count in token_counts.items()}

# Fonction pour calculer la probabilité de co-occurrence dans une fenêtre
def compute_joint_probability(token1, token2, token_list, window=3):
    count_joint = sum(1 for i in range(len(token_list) - window)
                      if token1 in token_list[i:i+window] and token2 in token_list[i:i+window])
    return count_joint / len(token_list) if count_joint > 0 else 0

# Fonction pour calculer l'information mutuelle (MI)
def compute_mi(token1, token2, prob_t1, prob_t2, joint_prob):
    EPSILON = 1e-10  # Éviter log(0)
    prob_t1, prob_t2, joint_prob = max(prob_t1, EPSILON), max(prob_t2, EPSILON), max(joint_prob, EPSILON)
    return joint_prob * np.log2(joint_prob / (prob_t1 * prob_t2))

# Fonction pour calculer l'entropie conjointe (H)
def compute_joint_entropy(joint_prob):
    return -joint_prob * np.log2(joint_prob) - (1 - joint_prob) * np.log2(1 - joint_prob) if 0 < joint_prob < 1 else 0

# Fonction pour calculer le PWI
def compute_pwi(mi, h):
    return mi - h

# Fonction parallèle pour traiter un lot de paires
def compute_pwi_for_pairs(pair_list, probabilities, tokens):
    results = []
    for t1, t2 in pair_list:
        joint_prob = compute_joint_probability(t1, t2, tokens)
        mi = compute_mi(t1, t2, probabilities.get(t1, 0), probabilities.get(t2, 0), joint_prob)
        h = compute_joint_entropy(joint_prob)
        pwi = compute_pwi(mi, h)
        results.append((t1, t2, pwi))
    return results

# Fonction pour traiter un chunk en multiprocessing
def process_chunk(args):
    chunk, probabilities, tokens = args
    return compute_pwi_for_pairs(chunk, probabilities, tokens)

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)  # Forcer le mode spawn
    # Charger les PDF
    pdf1_path = "info_retrieval.pdf"
    pdf2_path = "Vector.pdf"

    text1, text2 = extract_text_from_pdf(pdf1_path), extract_text_from_pdf(pdf2_path)

    # Prétraitement des textes
    tokens1, tokens2 = preprocess_text(text1), preprocess_text(text2)

    # Sélectionner les 1000 mots les plus fréquents de chaque article
    top_n = 1000
    top_tokens1 = [word for word, _ in Counter(tokens1).most_common(top_n)]
    top_tokens2 = [word for word, _ in Counter(tokens2).most_common(top_n)]

    # Calcul des probabilités des tokens
    total_tokens = len(tokens1) + len(tokens2)
    probabilities = compute_token_probabilities(tokens1 + tokens2, total_tokens)

    # Générer les paires entre article 1 et article 2
    token_pairs = list(itertools.product(top_tokens1, top_tokens2))  # N1 x N2

    # Nombre de cœurs CPU disponibles
    num_cores = min(multiprocessing.cpu_count(), 8)  # Utiliser max 8 cœurs

    # Diviser les paires en plusieurs groupes pour chaque thread
    chunk_size = len(token_pairs) // num_cores
    chunks = [token_pairs[i:i + chunk_size] for i in range(0, len(token_pairs), chunk_size)]

    # Préparer les arguments pour multiprocessing
    args_list = [(chunk, probabilities, tokens1 + tokens2) for chunk in chunks]

    # Exécution parallèle avec barre de progression
    print(f"Exécution parallèle sur {num_cores} cœurs pour {len(token_pairs)} paires...")
    with multiprocessing.Pool(num_cores) as pool:
        for _ in tqdm(pool.imap_unordered(process_chunk, args_list), total=len(args_list)):
            pass
    


    # Conversion en DataFrame
    flattened_results = [item for sublist in results for item in sublist]
    df_pwi = pd.DataFrame(flattened_results, columns=["Token1", "Token2", "PWI"])

    # Exporter la matrice sous format Excel
    output_excel_path = "pwi_parallel_optimized.xlsx"
    df_pwi.to_excel(output_excel_path, index=False)

    print(f"Les résultats ont été enregistrés dans {output_excel_path}")
