import subprocess
import time

# Wspólne ustawienia dla wszystkich eksperymentów
COMMON_ARGS = [
    "--img_examples", "100000",
    "--batch_size", "512",
    "--device", "cuda",
    "--wandb_project", "sae-hyperparam-sweep"
]

# Definicje eksperymentów
experiments = [
    # --- TopK SAE: Badamy wpływ liczby k (rzadkość sztywna) ---
    {
        "name": "TopK_k16",
        "args": ["--SAE", "TopK", "--k", "16"]
    },
    {
        "name": "TopK_k32",
        "args": ["--SAE", "TopK", "--k", "32"]
    },
    {
        "name": "TopK_k64",
        "args": ["--SAE", "TopK", "--k", "64"]
    },
    {
        "name": "TopK_k128",
        "args": ["--SAE", "TopK", "--k", "128"]
    },
    {
        "name": "TopK_k256",
        "args": ["--SAE", "TopK", "--k", "256"]
    },

    # --- ReLU SAE: Badamy wpływ współczynnika L1 (lambda) ---
    {
        "name": "ReLU_L1_1e-4",
        "args": ["--SAE", "ReLU", "--lambda_val", "0.0001"]
    },
    {
        "name": "ReLU_L1_3e-4",
        "args": ["--SAE", "ReLU", "--lambda_val", "0.0003"]
    },
    {
        "name": "ReLU_L1_6e-4",
        "args": ["--SAE", "ReLU", "--lambda_val", "0.0006"]
    },
    {
        "name": "ReLU_L1_1e-5",
        "args": ["--SAE", "ReLU", "--lambda_val", "0.00001"]
    },
    {
        "name": "ReLU_L1_1e-6",
        "args": ["--SAE", "ReLU", "--lambda_val", "0.000001"]
    },

    # --- SpaDE: Badamy inicjalizację skali i czy warto ją uczyć ---
    # 1. Baseline (Default Paper Init: 1/2d, Learned)
    {
        "name": "SpaDE_PaperDefault",
        "args": ["--SAE", "SpaDE"] 
    },
    # 2. Fixed Default (Sprawdzamy, czy uczenie parametru pomaga)
    {
        "name": "SpaDE_FixedDefault",
        "args": ["--SAE", "SpaDE", "--fix_lambda"]
    },
    # 3. High Init (Startujemy od bardzo "ostrego" sparsemaxa)
    {
        "name": "SpaDE_HighInit_10",
        "args": ["--SAE", "SpaDE", "--lambda_val", "0.5"]
    },
    # 4. Low Init (Startujemy od "miękkiego" sparsemaxa)
    {
        "name": "SpaDE_LowInit_0.1",
        "args": ["--SAE", "SpaDE", "--lambda_val", "0.1"]
    },
    # 5. Fixed High (Wymuszamy bardzo dużą rzadkość przez cały trening)
    {
        "name": "SpaDE_FixedHigh_20",
        "args": ["--SAE", "SpaDE", "--lambda_val", "0.01", "--fix_lambda"]
    },
]

def run():
    print(f"Planowane uruchomienie {len(experiments)} eksperymentów...")
    
    for i, exp in enumerate(experiments):
        print(f"\n==================================================")
        print(f"Rozpoczynanie eksperymentu {i+1}/{len(experiments)}: {exp['name']}")
        print(f"==================================================")
        
        # Konstrukcja komendy
        # Dodajemy unikalną nazwę runu do WandB, żeby łatwo je odróżnić
        cmd = ["python", "main.py"] + COMMON_ARGS + exp["args"] + ["--wandb_project", "sae-sweep-final"]
        
        # Opcjonalnie: dodaj nazwę runu jako tag lub argument, jeśli main.py to obsługuje
        # Tutaj polegamy na tym, że WandB i tak zapisze config
        
        try:
            # Uruchomienie procesu i czekanie na zakończenie
            subprocess.run(cmd, check=True)
            print(f"Eksperyment {exp['name']} zakończony sukcesem.")
        except subprocess.CalledProcessError as e:
            print(f"BŁĄD w eksperymencie {exp['name']}. Kod wyjścia: {e.returncode}")
            # Możesz dodać 'break' tutaj, jeśli chcesz przerwać po błędzie
            
        # Mała przerwa dla chłodzenia GPU / logowania WandB
        time.sleep(2)

if __name__ == "__main__":
    run()