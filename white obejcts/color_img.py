import os
import argparse
import itertools
from PIL import Image, ImageOps, ImageChops

def generuj_palete(k):
    """
    Generuje listę krotek RGB na podstawie liczby podziałów k.
    Dla k=3 zwraca 27 kombinacji (3*3*3).
    """
    if k < 2:
        # Fallback dla k=1 (zwraca tylko biały)
        return [(255, 255, 255)]
    
    # Wyliczamy wartości dla jednego kanału, np. dla k=3 -> [0, 127, 255]
    wartosci = [int(i * 255 / (k - 1)) for i in range(k)]
    
    # Tworzymy iloczyn kartezjański (wszystkie możliwe kombinacje R, G, B)
    kombinacje = list(itertools.product(wartosci, repeat=3))
    return kombinacje

def znajdz_pliki(folder_path):
    pliki = os.listdir(folder_path)
    maska = None
    obrazek = None
    
    for plik in pliki:
        lower_name = plik.lower()
        if lower_name.startswith("mask_") and lower_name.endswith(('.png', '.jpg', '.jpeg')):
            maska = plik
            break
            
    if maska:
        nazwa_bez_prefiksu = maska.replace("mask_", "")
        if nazwa_bez_prefiksu in pliki:
            obrazek = nazwa_bez_prefiksu
            
    return maska, obrazek

def przetwarzaj(glowny_folder, k):
    paleta = generuj_palete(k)
    print(f"--- START ---")
    print(f"Parametr k={k}. Wygenerowano {len(paleta)} kombinacji kolorystycznych.")
    
    for root, dirs, files in os.walk(glowny_folder):
        if root == glowny_folder:
            continue
            
        nazwa_maski, nazwa_obrazka = znajdz_pliki(root)
        
        if not nazwa_maski or not nazwa_obrazka:
            continue
            
        print(f"Przetwarzanie: {root} -> {nazwa_obrazka}")
        
        sciezka_img = os.path.join(root, nazwa_obrazka)
        sciezka_mask = os.path.join(root, nazwa_maski)
        
        try:
            img = Image.open(sciezka_img).convert("RGBA")
            mask = Image.open(sciezka_mask).convert("L")
            
            # Baza w szarości dla lepszego "chwytania" koloru
            img_gray = ImageOps.grayscale(img).convert("RGBA")
            
            output_dir = os.path.join(root, "output_k" + str(k))
            os.makedirs(output_dir, exist_ok=True)
            
            for r, g, b in paleta:
                # Kolor nakładki
                kolor_rgb = (r, g, b)
                warstwa_koloru = Image.new("RGBA", img.size, kolor_rgb + (255,))
                
                # Mnożenie (Multiply) - najlepsze dla jasnych/białych obiektów
                wymieszany = ImageChops.multiply(img_gray, warstwa_koloru)
                
                # Składanie finalne
                wynik = Image.composite(wymieszany, img, mask)
                
                # Nazwa pliku zawiera kod RGB, np. car_255_0_0.jpg
                nazwa_wynikowa = f"{os.path.splitext(nazwa_obrazka)[0]}_{r}_{g}_{b}.jpg"
                wynik.convert("RGB").save(os.path.join(output_dir, nazwa_wynikowa), quality=95)
                
        except Exception as e:
            print(f"Błąd w folderze {root}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automat do kolorowania zdjęć z maskami.")
    
    # Argument --path: ścieżka do folderu (domyślnie obecny folder kropka .)
    parser.add_argument("--path", type=str, default=".", help="Ścieżka do głównego folderu z podfolderami")
    
    # Argument --k: liczba podziałów na kanał
    parser.add_argument("--k", type=int, default=3, help="Liczba stopni na kanał koloru (k=3 daje 27 obrazków)")
    
    args = parser.parse_args()
    
    przetwarzaj(args.path, args.k)