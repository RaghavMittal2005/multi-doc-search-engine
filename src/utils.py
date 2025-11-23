from sklearn.datasets import fetch_20newsgroups
from pathlib import Path

def download_newsgroups_dataset(output_dir: str = "data/docs", max_docs: int = 200):
    """Download and save 20 newsgroups dataset"""
    print("Downloading 20 newsgroups dataset...")
    dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save first max_docs documents
    for i, (text, category) in enumerate(zip(dataset.data[:max_docs], dataset.target[:max_docs])):
        category_name = dataset.target_names[category]
        filename = f"doc_{i:03d}_{category_name}.txt"
        
        with open(output_path / filename, 'w', encoding='utf-8') as f:
            f.write(text)
    
    print(f"Saved {min(max_docs, len(dataset.data))} documents to {output_dir}")

if __name__ == "__main__":
    download_newsgroups_dataset()