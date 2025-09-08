# Project Notebooks

This repository contains a series of Jupyter Notebook files (.ipynb) that should be executed sequentially. Follow the numerical order in the filenames to ensure proper execution flow.

## File Execution Order

Run these notebooks in the following sequence:

1. `3.0_preprocessing.ipynb`
2. `3.1 TE_PID_PhiID_analysis.ipynb`
3. `4.1.1_TE_Network_Properties.ipynb`
4. `4.1.2_PID.ipynb`
5. `4.2.1_BCT.ipynb`
6. `4.2.2_Network_Similarity.ipynb`
7. `4.2.3_Topology.ipynb`
8. `4.3.1_Domirank_TE.ipynb`
9. `4.3.3_Temporal_Domirank.ipynb`

## Notebook Categories

- **Preprocessing** (`3.0_*`): Data preparation and cleaning
- **Information Theory Analysis** (`3.1_*, 4.1_*`): TE, PID, and PhiID analyses
- **Network Properties** (`4.2_*`): BCT, similarity, and topology analyses
- **Dominance Ranking** (`4.3_*`): Domirank and temporal dominance analyses


## Usage Notes

- Execute cells in each notebook sequentially from top to bottom
- Some notebooks may depend on outputs from previous notebooks
- Check that all cells complete successfully before moving to the next notebook
- Results and intermediate files are typically saved in the `results/` directory