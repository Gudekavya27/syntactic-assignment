# syntactic-assignment
# Identifying Key Entities in Recipe Data

This project focuses on training a Named Entity Recognition (NER) model using Conditional Random Fields (CRF) to extract key entities from recipe data. The model classifies words into predefined categories such as ingredients, quantities, and units, enabling the creation of a structured database of recipes and ingredients.

## Business Objective

The goal of this project is to build a structured database of recipes and ingredients that can be used to power advanced features in:
- Recipe management systems
- Dietary tracking apps
- E-commerce platforms

## Data Description

The input data is in JSON format, representing a **structured recipe ingredient list** with **NER labels**. Each record contains:
- `input`: A raw ingredient list from a recipe.
- `pos`: Corresponding part-of-speech (POS) tags or NER labels, identifying quantities, ingredients, and units.

### Example Data
```json
[
    {
        "input": "6 Karela Bitter Gourd Pavakkai Salt 1 Onion 3 tablespoon Gram flour besan 2 teaspoons Turmeric powder Haldi Red Chilli Cumin seeds Jeera Coriander Powder Dhania Amchur Dry Mango Sunflower Oil",
        "pos": "quantity ingredient ingredient ingredient ingredient ingredient quantity ingredient quantity unit ingredient ingredient ingredient quantity unit ingredient ingredient ingredient ingredient ingredient ingredient ingredient ingredient ingredient ingredient ingredient ingredient ingredient ingredient ingredient"
    }
]
```

## Project Workflow

### 1. Data Ingestion and Preparation
- Load the JSON data into a Pandas DataFrame.
- Tokenize the `input` and `pos` fields into `input_tokens` and `pos_tokens`.
- Validate the data by checking for mismatches in token lengths and cleaning invalid rows.

### 2. Exploratory Data Analysis (EDA)
- Flatten the token lists for analysis.
- Categorize tokens into `ingredients`, `units`, and `quantities`.
- Identify the top 10 most frequent items in each category and visualize them using bar plots.

### 3. Train-Validation Split
- Split the dataset into training (70%) and validation (30%) sets.

### 4. Feature Engineering
- Define token-level features for CRF training, including:
  - Core features (e.g., token, lemma, POS tag, shape).
  - Quantity and unit detection using regex patterns and keyword sets.
  - Contextual features (e.g., preceding and following tokens).

### 5. Model Training
- Train a CRF model using the training dataset with the following hyperparameters:
  - `algorithm='lbfgs'`
  - `c1=0.5` (L1 regularization)
  - `c2=1.0` (L2 regularization)
  - `max_iterations=100`
  - `all_possible_transitions=True`

### 6. Model Evaluation
- Evaluate the model on both training and validation datasets using:
  - Flat classification reports.
  - Confusion matrices.

### 7. Error Analysis
- Investigate misclassified samples in the validation dataset.
- Analyze errors by label type and provide insights.

### 8. Model Saving
- Save the trained CRF model as `trained_crf_model.pkl` for future use.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Install `sklearn-crfsuite`:
   ```bash
   pip install sklearn_crfsuite==0.5.0
   ```

## Usage

1. Open the Jupyter Notebook `Named_Entity_Recognition_Priyanuj_Misra_Assignment.ipynb`.
2. Follow the step-by-step instructions to preprocess the data, train the model, and evaluate its performance.

## Results

- The CRF model successfully identifies key entities in recipe data with high accuracy.
- Error analysis reveals areas for improvement, such as better handling of overlapping keywords and contextual cues.

## Insights

1. Some labels are misunderstood due to keyword overlap, requiring clearer contextual cues.
2. Confusion exists between ingredients and units, possibly due to POS tagging issues.
3. Certain token sequences create ambiguity, highlighting the need for improved feature engineering.

## Conclusion

This project demonstrates the effectiveness of CRF models for NER tasks in recipe data. The structured output can be used for various applications, including recipe management and dietary tracking.


