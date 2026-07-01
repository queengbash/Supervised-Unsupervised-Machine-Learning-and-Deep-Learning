# 3rd NELIREF Data Science and AI Summer School 2026

## Instructor (s)

Roland Abi

# Teaching Assistant (s)

Terkuma Saaondo

## Supervised Machine Learning, Unsupervised Machine Learning, and Deep Learning with Python

A comprehensive repository introducing the core machine learning and deep learning workflow with Python, using real, industry-standard tools: pandas, scikit-learn, and TensorFlow/Keras.

## Overview

These interactive notebooks are designed for learners who have completed the Python Programming and NLP/LLM/GenAI foundations and are ready to build real predictive models. Using a hands-on Diabetes Prediction case study as a template, you will practice the full pipeline: loading and cleaning real data, training supervised models (classification and regression), exploring unsupervised techniques (clustering and dimensionality reduction), and building a neural network with Keras.

## What You'll Learn

- **Supervised Learning** - Classification and regression using labeled data
- **Unsupervised Learning** - Clustering (K-Means) and dimensionality reduction (PCA) on unlabeled data
- **Deep Learning** - Building, training, and evaluating neural networks with TensorFlow/Keras
- **Data Preprocessing** - Handling missing values, encoding, scaling, and train/test splits
- **Feature Engineering** - Interaction terms, polynomial features, and feature selection
- **Model Evaluation** - Accuracy, precision, recall, F1, ROC-AUC, MAE, MSE, RMSE, R²
- **Overfitting Prevention** - Dropout, early stopping, and regularization
- **Model Explainability & Ethics** - An introduction to SHAP, LIME, fairness, and accountability

## Table of Contents

1. What is Supervised Learning?
2. Supervised vs. Other Learning Types
3. Task Types: Regression vs. Classification
4. Classical ML Algorithms
5. Data Preprocessing & Feature Engineering
6. Model Evaluation
7. Why Deep Learning? Neural Network Basics
8. DL in Python & Overfitting Prevention
9. Model Explainability & Ethics
10. Case Study: Diabetes Prediction
11. Practice Exercises

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Jupyter Notebook, JupyterLab, or Google Colab
- Completion of Task 1 (Python Basics) and Task 2 (NLP/LLM/GenAI) recommended
- Familiarity with pandas and numpy

### Installation

#### Option 1: Install Jupyter with Anaconda (Recommended for beginners)
```bash
# Download Anaconda from https://www.anaconda.com/download
# Follow the installation instructions for your operating system

# Then open Anaconda Navigator and launch Jupyter Notebook
```

#### Option 2: Install with pip
```bash
pip install jupyter notebook pandas numpy scikit-learn matplotlib seaborn tensorflow
```

#### Option 3: Use Google Colab (Recommended for this repository!)
1. Go to https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Select `ML_DL_Summer_School_2026.ipynb` or `ML_DL_Practice_Exercises.ipynb`
4. scikit-learn and TensorFlow are already installed - start learning immediately!

### Running the Notebooks

```bash
# Navigate to the notebook directory
cd path/to/notebook/directory

# Start Jupyter Notebook
jupyter notebook

# Open ML_DL_Summer_School_2026.ipynb first (Diabetes Prediction case study)
# Then open ML_DL_Practice_Exercises.ipynb (hands-on exercises)
```

## How to Use These Notebooks

1. **Read the explanations** - Each section starts with clear explanations of concepts
2. **Run the examples** - Execute code cells to see outputs immediately
3. **Experiment** - Modify code examples (try different models, hyperparameters, datasets)
4. **Take notes** - Add your own markdown cells with reflections after each section
5. **Complete exercises** - Work through the 6 practice exercises and self-check with the test cells

### Keyboard Shortcuts (Jupyter)

| Action | Shortcut |
|--------|----------|
| Run cell | `Shift + Enter` |
| Add cell below | `B` |
| Delete cell | `D D` (press D twice) |
| Switch to markdown | `M` |
| Switch to code | `Y` |
| Edit mode | `Enter` |
| Command mode | `Esc` |

## Practice Exercises

`ML_DL_Practice_Exercises.ipynb` includes 6 progressive exercises:

1. **Supervised Learning - Classification** - Train and evaluate a Logistic Regression classifier on the Iris dataset
2. **Supervised Learning - Regression** - Train and evaluate a Linear Regression model on a disease-progression dataset
3. **Unsupervised Learning - Clustering** - Group unlabeled data into clusters with K-Means
4. **Unsupervised Learning - Dimensionality Reduction** - Compress features down to 2D with PCA
5. **Deep Learning - Build Your First Neural Network** - Construct and train a Keras Sequential model
6. **Preventing Overfitting with Early Stopping** - Add an EarlyStopping callback to your training run

Each exercise has a self-check **Test Cell** that tells you immediately whether your solution passes.

**Solutions** - Try solving these yourself first! If you get stuck, ask for hints or revisit the matching section in `ML_DL_Summer_School_2026.ipynb`.

## Learning Path

```
Day 1-2: Supervised Learning basics → Classification → Regression
Day 2-3: Data Preprocessing → Feature Engineering → Model Evaluation
Day 4-5: Unsupervised Learning (Clustering & PCA) → Neural Network Basics
Day 6-7: Deep Learning in Python → Overfitting Prevention → Practice Exercises
Day 8-12: Build your own end-to-end ML project!
```

## Resources for Further Learning

### Documentation
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Keras Documentation](https://keras.io/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

### Interactive Learning
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Kaggle Learn](https://www.kaggle.com/learn) - free micro-courses on ML, pandas, and deep learning
- [fast.ai](https://www.fast.ai/) - practical deep learning courses

### Books
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron
- *Deep Learning with Python* by François Chollet

### YouTube Channels
- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) - clear explanations of ML/DL concepts
- [3Blue1Brown - Neural Networks series](https://www.youtube.com/@3blue1brown)
- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)

## Tips for Success

**Code along** - Type the examples yourself, don't just copy-paste  
**Understand the pipeline** - Data → Clean → Engineer Features → Train → Evaluate → Iterate  
**Debug systematically** - Read error messages carefully, especially shape-mismatch errors  
**Practice daily** - Even 15 minutes a day compounds over time  
**Use multiple metrics** - No single metric (accuracy, R², etc.) tells the whole story  
**Join communities** - Connect with other learners for support  

## Common ML/DL Patterns

### Train/Test Split & Scaling
```python
# Split first, then scale - fit the scaler ONLY on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # transform, not fit_transform!
```

### Commenting Best Practices
```python
# Use comments to explain WHY, not WHAT
# Good: L2 regularization (penalty='l2') helps prevent overfitting on small datasets
model = LogisticRegression(penalty='l2', C=1.0)

# Avoid: Create a logistic regression model
```

### Model Naming
```python
# Name variables by what they represent
model_lr = LogisticRegression()
model_nn = Sequential([...])
y_pred_lr = model_lr.predict(X_test)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn' or 'tensorflow'"
```bash
pip install scikit-learn tensorflow pandas matplotlib seaborn
# On Google Colab these are already installed
```

### My neural network's loss is NaN
- Make sure your features are scaled (`StandardScaler`) before training
- Try a smaller learning rate or fewer epochs
- Check for `NaN` or infinite values in your input data with `df.isna().sum()`

### KMeans gives a warning about `n_init`
```python
# Explicitly set n_init to silence the warning:
KMeans(n_clusters=3, random_state=42, n_init=10)
```

### My train/test shapes don't match
- Confirm you scaled `X_train` and `X_test` with the **same fitted** scaler (fit on train, transform on both)
- Print `X_train.shape` and `X_test.shape` to confirm the number of features matches

### Jupyter kernel keeps crashing
- Restart the kernel: Kernel → Restart
- Close other applications using RAM
- Update Jupyter: `pip install --upgrade jupyter`

## Contributing

Found an error? Have a suggestion?
1. Fork this repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This notebook is released under the MIT License - feel free to use, modify, and share!

## Getting Help

- **In the notebook** - Hover over error messages and read them carefully
- **Online** - Search your error message on Stack Overflow
- **Communities** - Join NELIREF's Telegram at https://t.me/neliref
- **Official Docs** - The scikit-learn and Keras documentation are excellent references

## What's Next After These Notebooks?

Once you've mastered supervised, unsupervised, and deep learning basics:

1. **Convolutional Neural Networks (CNNs)** - Image classification and computer vision
2. **Recurrent Neural Networks & Transformers** - Sequence and time-series data (ties back to Task 2's NLP concepts!)
3. **Hyperparameter Tuning** - GridSearchCV, RandomizedSearchCV, and Keras Tuner
4. **Model Deployment** - Serving models with Flask, FastAPI, or cloud platforms
5. **MLOps** - Experiment tracking, model versioning, and pipelines
6. **Advanced Ensemble Methods** - XGBoost, LightGBM, CatBoost
7. **Kaggle Competitions** - Apply everything you've learned to real competitions

## Author

Created for the 3rd NELIREF Data Science & AI Summer School 2026.

---

**Happy Coding!**

Questions? Issues? Reach out at contact@neliref.org, www.neliref.org, or on Telegram at https://t.me/neliref

Remember: Every expert was once a beginner. Keep practicing, stay curious, and enjoy the learning journey!
