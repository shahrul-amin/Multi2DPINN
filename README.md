# Multi2DPINN-EM

This repository contains the implementation of a Multi-stage 2D Physics-Informed Neural Network (PINN) for Electromagnetics (EM) modeling. 

## Project Structure

- **data/**: Contains datasets used for testing and training, including `.geo` (geometry) and `.mat` (material/features) files.
- **data_gen/**: Scripts for dataset generation and simulation (e.g., `simulate.py`, `autoGenGeo_correct.py`).
- **src/**: Core source code including definitions for the `first_stage` and `second_stage` networks.
- **run/**: Results and execution metadata for different stages and experiments.
- **train_afd.py**, **train_ss.py**: Training scripts for different models or stages.
- **plot_geometry.py**: Utility script to visualize the wire geometry from `.geo` files.

## Wire Visualization

The repository includes a script to visualize the wire geometries representing the segments. The visualization is constructed by reading the coordinate, length, and width parameters from OpenCASCADE format `.geo` files and plotting each segment.

### How to use

1. **Environment Setup**: Create a conda environment from `environment.yml`.
   ```bash
   conda env create -f environment.yml
   conda activate pinn
   ```

2. **Data Generation**: We used the scripts in the `data_gen/` folder to create a synthetic dataset for both first stage and second stage training.

3. **Train First Stage Model**: Train the supervised model to detect the stress in a single wire.
   ```bash
   python train_ss.py --data-path ./data/EMdataset_10seg_1n2/
   ```

4. **Train Second Stage Model**: After the first stage is done, train the second stage model to predict boundary conditions and AFD.
   ```bash
   python train_afd.py --data-path ./data/test_trees/ --model-path ./run/first_stage/EMdataset_10seg_1n2/
   ```

### First and Second Stage Visualization

- **First Stage**: In the first stage, the base structure components and initial segments are plotted to visualize the foundational topology.
  
  ![First Stage Wire Visualization](assets/wire_visualization.png)

- **Second Stage**: The second stage visualizes the complex, refined wire branching or full topology iterations derived from the models. The multi-stage representation visually distinguishes between parent and child segments in the wire progression. 
  
  ![Second Stage Wire Visualization](assets/wire_visualization2.png) 
