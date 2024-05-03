# Hidden-Markov-Based-Black-Litterman-Model
  In the tumultuous world of finance, where economic indicators often send mixed signals, making informed decisions becomes a Herculean task. As of 2022, the Federal Reserve's initiation of a series of rate hikes stands as a testament to the complexity of the financial landscape. Despite these measures, the persistence of high inflation alongside robust employment data presents a paradoxical scenario, underscoring the volatile nature of financial markets. In such a milieu, personal and professional judgments are prone to fluctuations, reflecting the overarching uncertainty that defines our contemporary economic environment.
This paper introduces a groundbreaking research project that leverages a Hidden Markov Model-based Black Litterman approach to navigate the quagmire of market signals. By constructing an automated pipeline, this research seeks to discern distinct market regimes and tailor portfolio compositions accordingly, aligning with investors' risk preferences. This initiative does not merely fulfill an academic purpose; it transcends scholarly boundaries to serve an ethical role, particularly salient in the wake of unforeseen financial cataclysms—often termed "black swan" events. The devastating financial toll of such crises, exemplified by the global financial meltdown of 2008 and the economic ramifications of the COVID-19 pandemic, highlights the acute need for this project.

Ultimately, this project aspires to bolster financial literacy, fostering a culture of resilience and savvy decision-making amid economic adversities. By peeling back the layers of market operations and elucidating the underpinnings of financial turmoil, it aims to empower society at large. In doing so, the research not only contributes to academic discourse but also holds the promise of making a profound impact on the practical world, guiding individuals towards a more informed, secure, and financially literate future.

To read the thesis: [Markov’s Insight: Unveiling Active Views with Hidden Markov Models for Enhanced Black-Litterman Portfolio Optimization](./HonorThesis.pdf)

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

### Structure of the Repository

The repository is structured into two primary directories:

- **`Data` Directory:** Contains all the input and output data necessary for the project.
- **`Code` Directory:** Includes Jupyter notebooks for running the model, with detailed comments in markdown cells, and custom libraries for model construction.

## Detailed File Descriptions

### Code Directory

- `EDA.ipynb`: An exploratory data analysis notebook for analyzing the distribution of the 30 industry portfolios. Includes additional tests for stationarity not covered in the main paper.

- `HMM.ipynb`: Demonstrates applying the Hidden Markov Model to the food industry, extending the algorithm to generate confidence and active views for all 30 industries, and implementing a simple trading strategy.

- `BlackLitterman.ipynb`: Walks through the construction, determination of hyperparameters, and training of the model on 30 industry portfolios. It includes a detailed example of a one-period portfolio trained without constraints on weights, using only a mean variance objective. The latter part of the notebook focuses on analyzing the returns (HMM-BL) of selected hyperparameter combinations, an essential step in the study's trial and error process not included in the final paper results.

- `BlackLitterman_with-Constraint.ipynb`: Contains similar components as the `BlackLitterman` notebook but adds constraints to the output HMM-BL portfolio weights and optimization over different objectives. This notebook concludes with an analysis of the best selected hyperparameter portfolios compared to baseline portfolios, presenting summary statistics included in the thesis.

- `HMM.py`: A Python class designed to support the `HMM` notebook, including functions for visualizing the results on each industry.

- `black_litterman.py` and `black_litterman_constrained.py`: Python classes to support Bayesian framework implementation in the `BlackLitterman` notebooks, with the latter extending the former to include weight constraints and optimization function selection.

- `toolkit.py`: A utility library offering basic and helper functions for analyzing financial data, focusing on data transformation and summary statistical analysis.

### Data Directory

Contains datasets such as `RF.csv` for US Treasury yields, `all_hyper_combo_portfolio_returns.csv` for HMM-BL portfolio returns across various hyperparameter combinations, and several other datasets detailing confidence, decisions, and expected returns based on simple and weighted trading algorithms from July 1993 to July 2023. It also includes the `Fama French 6 factors` dataset and 30 industry monthly returns datasets.

## Package Dependencies and Installation

Before running the models, ensure the following packages are installed with their specified versions:

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

Use Conda or Pip to install these packages.

## Reproducing the Analysis

For those interested in running the models and reproducing the paper's findings without engaging in the full breadth of additional analysis, the following streamlined process is recommended:

### Environment Setup

Ensure that all necessary packages are installed using either Conda or Pip. Reference the list of packages and their specific versions provided earlier in this document.

### Data Extraction and Preprocessing

1. **Begin with the `HMM.ipynb` Notebook:**
   - Open `HMM.ipynb` and import the required packages as outlined in the initial cell.
   - Navigate to the *Extract Values into CSV* section.
   - Execute the `extract_data` function with your chosen parameters for train size, window, omega, mu, and metrics. This step is crucial for generating the decision, view, and confidence files needed for subsequent analyses. For replication consistent with the paper, execute the following commands:

```python
decisions_simple, views_simple, confidence_simple = extract_data(train_size=2/3, window=200, omega=0.6, mu=0.5, metrics='simple')
decisions_weighted, views_weighted, confidence_weighted = extract_data(train_size=2/3, window=200, omega=0.6, mu=0.5, metrics='weighted')

# Save the extracted data to CSV files for further use.
confidence_simple.to_csv('../Data/confidence_simple.csv')
decisions_simple.to_csv('../Data/decisions_simple.csv')
views_simple.to_csv('../Data/views_simple.csv')
confidence_weighted.to_csv('../Data/confidence_weighted.csv')
decisions_weighted.to_csv('../Data/decisions_weighted.csv')
views_weighted.to_csv('../Data/views_weighted.csv')
```

2. **Proceed to the `BlackLitterman_with-Constraint.ipynb` Notebook:**
   - Import the necessary packages and datasets as instructed at the beginning of the notebook.
   - Follow the steps outlined in the *Building Trailing Window* section to set up your analysis environment.
   - Continue through the notebook, executing all cells under the *Hyperparameter Selection* section to generate portfolio returns across all hyperparameter combinations.
   - Analyze and compare the best hyperparameter portfolios with baseline models in the *Visualization on Best Hyperparameter Models* section.
   - In the *Adjusting the Composition of a Portfolio in Graphing* section, generate backtesting results and metrics for both baseline portfolios and HMM-BL portfolios.

### Reproducing Paper Figures and Tables

To recreate specific elements from the paper:

- **Tables 3 and 4:** These can be generated by running the relevant code in the `BlackLitterman_with-Constraint.ipynb` notebook, specifically within the *Adjusting the Composition of a Portfolio in Graphing* section.
- **Figures 1-4:** Execute all cells in `EDA.ipynb` up to the point marked “Trying Food First” to generate these figures.
- **Figure 8:** In `BlackLitterman_with-Constraint.ipynb`, reproduce this figure by executing cells up to the *Construct a Class for Each Single Period* section.
- **Figures 9 and 10:** These figures can be reproduced by following the instructions in the *Adjusting the Composition of a Portfolio in Graphing* section of the `BlackLitterman_with-Constraint.ipynb` notebook.

## Contributing

Contributions are what make the open-source community an incredibly enriching place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would enhance this project, we encourage you to take action by participating in its development. Here’s how you can do so:

1. **Fork the Project:** Navigate to the GitHub repository and use the fork button to create your own copy of the project.
2. **Create Your Feature Branch:** Checkout a new branch for your feature by running `git checkout -b newEdits/SuggestedImprovements`.
3. **Commit Your Changes:** Once your feature is ready, commit your changes with a descriptive message, e.g., `git commit -m 'Add some suggested improvements'`.
4. **Push to the Branch:** Upload your feature branch to your forked repository with `git push origin newEdits/SuggestedImprovements`.
5. **Open a Pull Request:** From your forked repository, initiate a pull request. This is your opportunity to share your contribution with the project.

Feel free to open an issue with the tag "enhancement" if you have suggestions but are not ready to make a direct contribution. And don't forget to give the project a star if you find it helpful!

## Contact

Should you have any questions or wish to discuss this project further, please feel free to reach out:

- **Polly Pu** - pollypqt1101@berkeley.edu

Your interest and contributions are what continue to drive the remarkable progress and innovation within the open-source community. Thank you for your support and engagement!

---
