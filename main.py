import os
import pandas as pd
from Bio import SeqIO

tsv_file = "BindingDB_All.tsv"
fasta_file = "BindingDBTargetSequences.fasta"
cache_file = "beta_lactamase_tem_data.pkl"  # cached filtered data
chunksize = 100_000

def load_filtered_data():
    if os.path.exists(cache_file):
        print(f"Loading cached filtered data from '{cache_file}'...")
        return pd.read_pickle(cache_file)
    else:
        print("No cache found. Processing TSV to filter data...")
        use_cols = ['Target Name', 'Ligand SMILES', 'Kd (nM)', 'IC50 (nM)', 'Ki (nM)']
        filtered_chunks = []

        for i, chunk in enumerate(pd.read_csv(tsv_file, sep='\t', usecols=use_cols, chunksize=chunksize, low_memory=False)):
            tem_chunk = chunk[chunk['Target Name'] == 'Beta-lactamase TEM']
            if not tem_chunk.empty:
                filtered_chunks.append(tem_chunk)
            print(f"Processed chunk {i + 1}")

        if filtered_chunks:
            tem_df = pd.concat(filtered_chunks, ignore_index=True)
            tem_df.to_pickle(cache_file)  # save cache
            print(f"Saved filtered data cache to '{cache_file}'.")
            return tem_df
        else:
            print("No entries found for 'Beta-lactamase TEM'.")
            return pd.DataFrame()  # empty dataframe

def search_fasta_for_tem():
    print("\nSearching FASTA for sequences related to beta-lactamase TEM...")
    found = False
    for record in SeqIO.parse(fasta_file, "fasta"):
        desc_lower = record.description.lower()
        if "beta-lactamase" in desc_lower or "tem" in desc_lower:
            print(f"Found sequence: {record.id}")
            print(f"Description: {record.description}")
            print(f"Sequence length: {len(record.seq)} amino acids")
            print(f"First 100 amino acids:\n{record.seq[:100]}")
            found = True
            break
    if not found:
        print("No matching beta-lactamase TEM sequences found in FASTA.")

def main():
    tem_df = load_filtered_data()
    if not tem_df.empty:
        print(f"\nTotal 'Beta-lactamase TEM' binding entries: {len(tem_df)}")
        print(tem_df.head())
    search_fasta_for_tem()

if __name__ == "__main__":
    main()
