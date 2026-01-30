import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Plot global correlations from CSV summary.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the summary CSV file.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the plot.")
    parser.add_argument("--category", type=str, default=None, help="Filter by specific category (optional).")
    args = parser.parse_args()

    # 1. Wczytanie danych
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        return

    df = pd.read_csv(args.csv_file)

    # Filtrowanie po kategorii, jeśli podano
    if args.category:
        df = df[df['Category'] == args.category]

    # 2. Mapowanie nazw metryk na ładniejsze (zgodne z Twoim obrazkiem)
    # lch_euclidean -> HCL
    # lab -> CIELAB
    # rgb -> RGB
    metric_map = {
        'rgb': 'RGB',
        'lab': 'CIELAB',
        'lch_euclidean': 'HCL',
        'lch': 'LCH (Circular)'
    }
    df['Color Space'] = df['Color_Metric'].map(metric_map).fillna(df['Color_Metric'])

    # 3. Ustalenie kolejności modeli na osi X
    # Możesz dodać/usunąć modele z tej listy, aby zmienić kolejność
    order = ["Raw", "TopK", "ReLU", "SpaDE", "SzpaDeLDiag", "SzpaDeL", "SzpaDeLRank1", "SzpaDeLMultiHead"]
    # Filtrujemy tylko te, które faktycznie są w CSV
    present_models = [m for m in order if m in df['SAE_Type'].unique()]
    # Dodajemy ewentualne inne modele, których nie ma na liście 'order'
    remaining = [m for m in df['SAE_Type'].unique() if m not in present_models]
    final_order = present_models + remaining

    # 4. Rysowanie wykresu
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    # Barplot: X=Model, Y=Korelacja, Hue=Metryka Koloru
    ax = sns.barplot(
        data=df,
        x='SAE_Type',
        y='Global_Spearman',
        hue='Color Space',
        order=final_order,
        palette="tab10",
        edgecolor="black",
        linewidth=0.5
    )

    # 5. Dodanie statystyk (wartości) nad słupkami
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, rotation=90)

    # 6. Kosmetyka wykresu
    plt.title(f"Global Spearman Correlation by Model and Color Space ({args.category if args.category else 'All'})", fontsize=16)
    plt.ylabel("Global Spearman Correlation", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Color Space", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axhline(0, color='black', linewidth=0.8) # Linia pozioma na 0
    plt.tight_layout()

    # 7. Zapis
    os.makedirs(args.output_dir, exist_ok=True)
    filename = "global_correlation_summary.png"
    if args.category:
        filename = f"global_correlation_summary_{args.category}.png"
    
    save_path = os.path.join(args.output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main()