## Abstract
This project explores different machine learning models for predicting whether there will be a low or high demand on shared bicycles in Washington DC. The explored models include Logistic Regression, K-nearest neighbors, Random forest, XGBoost and CatBoost. Hyperparameter tuning was done on all the models and after evaluating them on a validation-set with recall as focus metric. Finally, the random forest model was selected to be used in production.

## How to run

**With Docker:**

```bash
docker compose up
```

**Manually:**

Install requirements:

```bash
pip install -r requirements.txt
```

Run the model api:

```bash
uvicorn app.api.model-api:app --host 0.0.0.0 --port 80
```

Launch the frontend:

```bash
streamlit run app/App.py --server.port=8501 --server.address=0.0.0.0
```

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
