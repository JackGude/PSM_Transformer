# src/pca_ground_truth.py
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm

def generate_ground_truth_pca(data_path, vocab_path, output_dir, make_gif=False):
    print("--- Generating Ground Truth 3D PCA ---")
    
    # 1. Load Data & Vocab
    df = pd.read_csv(data_path)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        psv_vocab = json.load(f)
    
    psv_map = {shift: i for i, shift in enumerate(psv_vocab)}
    psv_size = len(psv_vocab)
    languages = ['French', 'Spanish', 'Italian', 'Romanian']
    
    # 2. Aggregate Vectors directly from CSV
    lang_sums = {l: np.zeros(psv_size) for l in languages}
    lang_counts = {l: 0 for l in languages}
    
    print("Aggregating Ground Truth Vectors...")
    for lang in languages:
        col_name = f"PSV_{lang}"
        if col_name not in df.columns:
            print(f"Skipping {lang} (Column not found)")
            continue
            
        for item in tqdm(df[col_name].dropna(), desc=lang):
            if item == "[]": continue
            try:
                shifts = json.loads(item)
                for shift in shifts:
                    if shift in psv_map:
                        idx = psv_map[shift]
                        lang_sums[lang][idx] += 1.0 # Add actual occurrence
                lang_counts[lang] += 1
            except: pass

    # 3. Prepare PCA
    # Normalize: We want the Average Vector (Probability per word)
    vectors = [np.zeros(psv_size)] # Origin (Latin)
    labels = ["Latin (Origin)"]
    
    color_map = {
        'French': '#e74c3c', 'Spanish': '#f39c12', 
        'Italian': '#27ae60', 'Romanian': '#2980b9', 
        'Latin (Origin)': '#2c3e50'
    }
    
    for lang in languages:
        if lang_counts[lang] > 0:
            # Average vector = Total Counts / Total Words
            vec = lang_sums[lang] / lang_counts[lang] 
            vectors.append(vec)
            labels.append(lang)

    X = np.array(vectors)
    
    # 4. Run PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    var = pca.explained_variance_ratio_
    total_var = np.sum(var)
    
    print(f"\n--- RESULTS ---")
    print(f"Explained Variance: PC1={var[0]:.1%}, PC2={var[1]:.1%}, PC3={var[2]:.1%}")
    print(f"Total Variance Explained: {total_var:.1%}")
    
    # 5. Plotting (Same pretty style)
    sns.set_style("white")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        X_pca[0, 0], X_pca[0, 1], X_pca[0, 2],
        c=color_map['Latin (Origin)'], s=600, marker='*',
        edgecolors='white', linewidth=2, label='Latin (Origin)', depthshade=False
    )

    for i in range(1, len(labels)):
        lang = labels[i]
        ax.scatter(
            X_pca[i, 0], X_pca[i, 1], X_pca[i, 2],
            c=color_map.get(lang, 'gray'), s=500,
            edgecolors='white', linewidth=2, label=lang, depthshade=False
        )
        ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], f" {lang}", fontsize=14, fontweight='bold')

    origin = X_pca[0]
    for i in range(1, len(labels)):
        ax.plot(
            [origin[0], X_pca[i, 0]],
            [origin[1], X_pca[i, 1]],
            [origin[2], X_pca[i, 2]],
            color=color_map.get(labels[i], 'gray'), linestyle='--', alpha=0.3
        )
        ax.plot(
            [X_pca[i, 0], X_pca[i, 0]],
            [X_pca[i, 1], X_pca[i, 1]],
            [0, X_pca[i, 2]],
            color='gray', linestyle=':', alpha=0.5
        )

    ax.set_title("Ground Truth Drift Map (Raw Data)", fontsize=20)
    ax.set_xlabel(f"PC1 (Variance) {var[0]:.1%}")
    ax.set_ylabel(f"PC2 (Variance) {var[1]:.1%}")
    ax.set_zlabel(f"PC3 (Variance) {var[2]:.1%}")
    ax.view_init(elev=50, azim=230)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "pca_ground_truth.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    if make_gif:
        print("Rendering rotation GIF...")
        def update(angle):
            ax.view_init(elev=50, azim=230 + angle)
            return ax,

        frames = np.linspace(0, 360, num=450)
        gif = animation.FuncAnimation(
            fig, update, frames=frames, interval=50, blit=False
        )
        gif_path = os.path.join(output_dir, "pca_ground_truth.gif")
        gif.save(gif_path, writer=animation.PillowWriter(fps=45))
        print(f"GIF saved to {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/test_dataset.csv")
    parser.add_argument("--vocab", default="data/processed/psv_vocabulary.json")
    parser.add_argument("--out", default="results/visuals")
    parser.add_argument("--gif", action="store_true", help="Export rotating GIF in addition to PNG")
    args = parser.parse_args()

    generate_ground_truth_pca(args.data, args.vocab, args.out, make_gif=args.gif)