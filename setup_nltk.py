import nltk

nltk.download('punkt')
nltk.download('stopwords')

print("Téléchargement des ressources NLTK terminé !")
import multiprocessing

if __name__ == "__main__":
    num_cores = min(multiprocessing.cpu_count(), 8)
    print(f"Nombre de cœurs détectés : {multiprocessing.cpu_count()}")
    print(f"Nombre de cœurs utilisés pour multiprocessing : {num_cores}")
