# Matrix Completion using Denoising AutoEncoder

This project implements matrix completion using neural networks, with a focus on Denoising AutoEncoder (DAE) for collaborative filtering. The repository includes Python code for single-layer and deep DAE matrix completion, along with dataset normalization scripts for Jester and MovieLens datasets.

## Report

The repository includes a comprehensive report that covers a state of the art and an explanation of the work realized using DAE matrix completion.

## Usage

1. Install the required packages listed below.
2. Select the desired dataset and ensure the correct path.
3. Adjust parameters if necessary.
4. Run the code.

## Input and Output

Input: The matrix to complete in .xlsx format.
Output: Trained AutoEncoder with plots of loss evolution during training and validation.

## Data

The "Data" folder contains:
- "jester-data-1.xls": Jester dataset, accessible at https://goldberg.berkeley.edu/jester-data/
- "jester_normalized.xlsx": Normalized version of "jester-data-1.xls" with values in the range [0,1].
- "movielens.csv": Small MovieLens latest dataset, accessible at https://grouplens.org/datasets/movielens/latest/
- "movielens_normalized.xlsx": Normalized version of "movielens.csv" with values in the range [0,1].
