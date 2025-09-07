# Customer Segmentation

Unsupervised customer segmentation using clustering to discover actionable customer groups from tabular data. The notebook explores K-Means (with elbow and silhouette analysis), PCA visualization, and a DBSCAN baseline.

## Project Goals
- Clean and preprocess raw customer data
- Explore distributions and missing values
- Cluster customers and visualize segments
- Evaluate cluster quality (silhouette score)
- Prepare a reusable preprocessing pipeline

## Data (columns)
Typical fields include:
- Categorical: Gender, Ever_Married, Graduated, Profession, Spending_Score, Var_1 (renamed to SubSegment), Segmentation (label)
- Numerical: Age, Work_Experience, Family_Size
- ID (dropped)

Files:
- train.csv
- test.csv

## Approach
1. EDA: head/info/describe and basic distribution plots
2. Cleaning:
   - Drop ID
   - Train/validation split (stratified on Segmentation)
   - Impute categorical with mode and Family_Size with median
   - Rename Var_1 → SubSegment
3. Encoding:
   - Label encode selected categorical columns
4. Scaling:
   - Standardize numerical features (Work_Experience, Age, Family_Size)
5. Clustering:
   - K-Means: elbow method (k=1..10), select k=4
   - Silhouette score for quality check
   - DBSCAN as an alternative baseline
6. Visualization:
   - PCA to 2D and scatter plots colored by clusters
   - Optional composite label: Cluster + SubSegment
7. Pipeline (planned):
   - ColumnTransformer/Pipeline to replicate preprocessing on validation and test data

## Results & Visualizations
- Elbow curve to guide k choice
- 2D PCA scatter plots for K-Means and DBSCAN clusters
- Silhouette score printed for chosen k

## Repository Structure
- notebook.ipynb — main walkthrough for EDA, preprocessing, clustering, and plots
- train.csv, test.csv — input data
- README.md — project description

## How to Run
On Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter pandas numpy scikit-learn seaborn matplotlib
jupyter notebook notebook.ipynb
```

## Notes and Next Steps
- Consider one-hot encoding for categorical variables before clustering
- Finalize a ColumnTransformer + Pipeline to ensure consistent preprocessing for validation/test
- Try other clustering methods (Gaussian Mixture, Agglomerative) and tune DBSCAN (eps, min_samples)
- Add metrics per cluster (size, feature means) and business-friendly segment profiles
- Save artifacts (scaler, encoder, model) for deployment/reuse


## Acknowledgments
Built with pandas, scikit-learn,
