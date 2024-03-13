# Hidden-Markov-Based-Black-Litterman-Model
  In the tumultuous world of finance, where economic indicators often send mixed signals, making informed decisions becomes a Herculean task. As of 2022, the Federal Reserve's initiation of a series of rate hikes stands as a testament to the complexity of the financial landscape. Despite these measures, the persistence of high inflation alongside robust employment data presents a paradoxical scenario, underscoring the volatile nature of financial markets. In such a milieu, personal and professional judgments are prone to fluctuations, reflecting the overarching uncertainty that defines our contemporary economic environment.
This paper introduces a groundbreaking research project that leverages a Hidden Markov Model-based Black Litterman approach to navigate the quagmire of market signals. By constructing an automated pipeline, this research seeks to discern distinct market regimes and tailor portfolio compositions accordingly, aligning with investors' risk preferences. This initiative does not merely fulfill an academic purpose; it transcends scholarly boundaries to serve an ethical role, particularly salient in the wake of unforeseen financial cataclysms—often termed "black swan" events. The devastating financial toll of such crises, exemplified by the global financial meltdown of 2008 and the economic ramifications of the COVID-19 pandemic, highlights the acute need for this project.

Ultimately, this project aspires to bolster financial literacy, fostering a culture of resilience and savvy decision-making amid economic adversities. By peeling back the layers of market operations and elucidating the underpinnings of financial turmoil, it aims to empower society at large. In doing so, the research not only contributes to academic discourse but also holds the promise of making a profound impact on the practical world, guiding individuals towards a more informed, secure, and financially literate future.

---

# Reproducing the Hidden Markov Based Black-Litterman Model Project

This guide provides detailed instructions for reproducing the analysis and results of the Hidden Markov Based Black-Litterman Model project.

## Getting Started

### Cloning the Repository

To get started with the project, first clone the repository using git:

```bash
git clone https://github.com/PollyPu23/Hidden-Markov-Based-Black-Litterman-Model.git
```

Alternatively, you can download the repository directly from the GitHub page:

[Hidden Markov Based Black-Litterman Model Repository](https://github.com/PollyPu23/Hidden-Markov-Based-Black-Litterman-Model.git)

### Repository Structure

The repository is organized into two main directories:

- `Data`: Contains all input and output data.
- `Code`: Contains Jupyter notebooks and self-created libraries for model building and analysis.

#### Code Directory Contents

- `EDA.ipynb`: Performs early-stage Exploratory Data Analysis (EDA) on 30 industry portfolios, including tests for stationarity.
- `HMM.ipynb`: Applies the Hidden Markov Model (HMM) and a simple trading strategy to the food industry, extending to all 30 industries with generated confidence and active views.
- `BlackLitterman.ipynb`: Demonstrates the construction and hyperparameter tuning of the Black-Litterman model on 30 industry portfolios, including detailed analysis and evaluation metrics.
- `BlackLitterman_with-Constraint.ipynb`: Similar to `BlackLitterman.ipynb` but includes constraints on portfolio weights and optimization objectives, with additional analysis on hyperparameter selection.
- `HMM.py`: A Python class supporting HMM analysis with visualization functions.
- `black_litterman.py` & `black_litterman_constrained.py`: Python classes for performing the Bayesian framework and optimization with constraints in the Black-Litterman model.
- `toolkit.py`: Provides helper functions for financial data analysis.

#### Data Directory Contents

Contains datasets such as:
- Monthly US Treasury yields (`RF.csv`).
- Hyperparameter combinations for HMM-BL portfolio returns (`all_hyper_combo_portfolio_returns.csv`).
- Confidence and decisions datasets for simple and weighted trading algorithms.
- The `Fama French 6 factors` dataset and `30 industry` datasets for various analyses.

### Required Packages and Versions

Ensure the following packages are installed with their respective versions:

- pandas 2.0.3
- numpy 1.24.3
- yfinance 0.2.35
- seaborn 0.12.2
- matplotlib 3.7.2
- scipy 1.11.1
- python-dateutil 2.8.2
- scikit-learn 1.3.0
- hmmlearn 0.3.2
- statsmodels 0.14.0

Install these using Conda or Pip as appropriate for your environment.

## Workflow

### Running the Model

1. **Initial Data Extraction:**
   - Open `HMM.ipynb` and run the section "Extract Values into CSV" to generate decision, view, and confidence files. Example command:
     ```python
     decisions_simple, views_simple, confidence_simple = extract_data(train_size=2/3, window=200, omega=0.6, mu=0.5, metrics='simple')
     # Save to CSV as demonstrated in the notebook.
     ```

2. **Model Training and Analysis:**
   - In `BlackLitterman_with-Constraint.ipynb`, import necessary packages and datasets, then proceed through the notebook to train the model, select hyperparameters, and analyze outcomes.

### Reproducing Paper Elements

- **Tables 3 and 4:** Follow instructions in `BlackLitterman_with-Constraint.ipynb` under "Adjusting the Composition of a Portfolio in Graphing" section.
- **Figures 1-4, 8-10:** Utilize `EDA.ipynb` and `BlackLitterman_with-Constraint.ipynb` as specified to generate relevant figures and tables.

This comprehensive guide should facilitate the reproduction of all analyses and results from the Hidden Markov Based Black-Litterman Model project.

---
