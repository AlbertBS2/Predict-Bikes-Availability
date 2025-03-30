## Abstract
This project explores different machine learning models for predicting whether there will be a low or high demand on shared bicycles in Washington DC. The explored models include Logistic Regression, K-nearest neighbors, Random forest, XGBoost and CatBoost. Hyperparameter tuning was done on all the models and after evaluating them on a validation-set with recall as focus metric, random forest was deemed the winner. The random forest model achieved a recall score of 64% and overall accuracy of 88%. The Random Forest model was used to obtain predictions on the final unseen test set.

## Directory Structure

- `data/`: Contains all CSV files used and generated during the project.
- `libs/`: Includes libraries with functions used across other files.
- `models/`: Jupyter notebooks for training, testing, and optimizing all models.
- `output/best_params/`: JSON files containing the best parameters obtained for each model.
- `data_analysis.ipynb`: Notebook for data exploration and visualization.
- `main.ipynb`: Notebook for generating final results and graphs.

---

## Reproducing the Results

Follow these steps to reproduce the results:

1. **Install Dependencies**  
   - Ensure you have the required libraries by installing them from `requirements.txt`.

2. **Data Analysis**
   - Open `data_analysis.ipynb`.
   - Run all cells to view the results and graphs.

3. **Model Training and Optimization**
   - Open each notebook in the `models/` directory.
   - Run all cells in each notebook. The optimal parameters for each model will be saved in the `models/best_params/` directory.

4. **Final Results**
   - Open `main.ipynb`.
   - Run all cells to see the final results and graphs.
