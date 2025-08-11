# Beta-lactamase TEM Ligand Binding Prediction

## Project Overview

This project aims to develop a computational model that predicts the binding affinity of small molecule ligands to the bacterial enzyme **Beta-lactamase TEM**. Beta-lactamase enzymes are a major cause of antibiotic resistance by breaking down beta-lactam antibiotics, making infections harder to treat. Accurate prediction of ligand binding can accelerate drug discovery and help design new inhibitors to combat resistant bacteria.

---

## Why This Project Matters

- **Antibiotic resistance is a critical global health threat.** Beta-lactamases like TEM contribute heavily to resistance against penicillins and cephalosporins.
- Developing new inhibitors targeting Beta-lactamase TEM can restore the effectiveness of existing antibiotics.
- Computational predictions drastically reduce time and cost compared to experimental assays.
- This project leverages open-access biochemical binding data to build an AI-powered prediction tool, demonstrating the impact of computer science on real-world biomedical challenges.

---

## Data Sources

- **BindingDB Dataset:** Contains ligand structures (SMILES) and binding affinity measurements (Kd, IC50, Ki) for Beta-lactamase TEM.  
- **FASTA Protein Sequences:** Amino acid sequences of Beta-lactamase TEM extracted from BindingDB resources.  
- *(Optional future additions)* Protein 3D structures (PDB), resistance gene variant sequences (CARD database).

---

## How It Works

1. **Data Loading and Filtering:**  
   The BindingDB dataset (~500 MB TSV file) is loaded in chunks and filtered for entries related to Beta-lactamase TEM to reduce size and focus on relevant data.

2. **Data Caching:**  
   Filtered ligand and binding data are cached locally to speed up subsequent runs and ease development.

3. **Sequence Search:**  
   Protein sequences relevant to Beta-lactamase TEM are extracted from FASTA files for potential feature extraction.

4. **Feature Extraction and Model Building:**  
   Ligand molecular features are generated from SMILES strings using cheminformatics tools (e.g., RDKit). Protein features may be integrated as the project evolves.

5. **Machine Learning:**  
   A regression or classification model is trained to predict ligand binding affinity, using curated BindingDB data as ground truth.

---

## Getting Started

### Requirements

- Python 3.8+  
- Pandas  
- Biopython  
- RDKit (for chemical feature extraction)  
- Scikit-learn or PyTorch/TensorFlow (for modeling)

### Usage

1. Download and unzip the **BindingDB_All_202508.tsv** and **BindingDBTargetSequences.fasta** files from [BindingDB](https://www.bindingdb.org/).  
2. Run the provided script to load, filter, and cache Beta-lactamase TEM ligand binding data.  
3. Extend with feature engineering and model training as needed.

---

## Future Work

- Integrate 3D structural data for docking-based features.  
- Explore mutation impacts by including variant sequences from resistance databases.  
- Develop a web app for interactive prediction and visualization.  
- Collaborate with experimentalists to validate model predictions.

---

## Contact

Created by IK_Knight / Isaaq K
For questions or collaboration, please reach out via mohammadIsaaqK@gmail.com.
