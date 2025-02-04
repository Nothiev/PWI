import multiprocessing

def test_function(x):
    return x * x

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # Important sous Windows
    
    num_cores = min(multiprocessing.cpu_count(), 8)
    print(f"Test multiprocessing sur {num_cores} cœurs...")

    with multiprocessing.Pool(num_cores) as pool:
        results = pool.map(test_function, range(10))
    
    print("Résultat du test multiprocessing :", results)
