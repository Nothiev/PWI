import multiprocessing
from tqdm import tqdm

def test_function(x):
    return x * x

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # Important sous Windows

    num_cores = min(multiprocessing.cpu_count(), 8)
    print(f"Test multiprocessing sur {num_cores} cœurs...")

    with multiprocessing.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(test_function, range(100)), total=100))

    print("Multiprocessing fonctionne correctement :", results[:10])
