# Fama–French 3-Factor Model Playground

![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue)

## Overview

The **Fama–French 3-Factor Model Playground** is an interactive educational tool built with [Streamlit](https://streamlit.io/) to help students and finance enthusiasts understand the core concepts behind the Fama–French 3-Factor Model. This model is an extension of the Capital Asset Pricing Model (CAPM) and explains asset returns by considering three main factors:

- **Market Factor (Mkt-RF):** Excess return of the market (Market Return minus Risk-Free Rate)
- **Size Factor (SMB):** Return difference between small and large companies ("Small Minus Big")
- **Value Factor (HML):** Return difference between value stocks and growth stocks ("High Minus Low")

The tool allows you to simulate asset returns, adjust factor loadings, and experiment with different empirical labs to see how changes in parameters can affect the detection of "manager skill" (alpha).

## Features

- **Interactive Controls:**  
  Easily adjust parameters (alphas, betas, risk-free rate, and noise) using intuitive sliders.
  
- **Empirical Labs:**  
  Three labs help illustrate:
  - **Lab 1:** Detecting Manager Skill (small extra return with minimized noise)
  - **Lab 2:** Factor Timing Strategy (increased exposure to the size factor)
  - **Lab 3:** Crisis Period Analysis (simulating market stress scenarios)
  
- **Step-by-Step Guidance:**  
  Detailed explanations in separate tabs covering the theoretical foundation, a practical guide, and empirical examples.
  
- **Visualizations:**  
  Regression results, factor exposure analysis, and comparisons between predicted and actual returns are visualized using Matplotlib.

## Getting Started

### Prerequisites

Make sure you have the following installed:
- Python 3.7+
- [Streamlit](https://docs.streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Statsmodels](https://www.statsmodels.org/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/fama-french-playground.git
   cd fama-french-playground
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` file, create one with:

   ```text
   streamlit
   numpy
   pandas
   matplotlib
   statsmodels
   ```

### Running the App

To launch the app, run:

```bash
streamlit run your_app_file.py
```

Replace `your_app_file.py` with the name of the Python file containing the code (e.g., `fama_french_playground.py`).

## How It Works

### Model Overview

The app simulates asset returns using the following equation:

$$
\text{Return} - \text{Risk-Free} = \alpha + \beta_{Mkt}(Mkt-RF) + \beta_{SMB}(SMB) + \beta_{HML}(HML) + \epsilon
$$

- **α (Alpha):** The extra return not explained by the factors.
- **β (Beta):** Sensitivities to the factors.
- **ε (Noise):** Random idiosyncratic risk.

### Interactive Tabs

1. **Interactive Analysis:**  
   View regression results, examine estimated factor sensitivities, and compare predicted versus actual returns.
   
2. **Theoretical Foundation:**  
   Understand the model's components and the economic rationale behind each factor.
   
3. **Step-by-Step Guide:**  
   Follow a simple tutorial on the model and how to interpret the regression outputs.
   
4. **Empirical Labs:**  
   Experiment with different setups:
   - **Lab 1:** Test if a small alpha (0.5% per month) is statistically significant when noise is low (1% per month).
   - **Lab 2:** Explore factor timing by increasing the SMB beta.
   - **Lab 3:** Simulate a market crisis with reduced market sensitivity and negative exposures.
   
5. **Factor Investing Basics:**  
   Learn about the basics of factor investing and the challenges faced in real-world applications.

## Pedagogical Approach

This tool is designed for students new to empirical asset pricing:
- **Clear Explanations:** Each parameter and equation is explained in simple terms.
- **Visual Learning:** Graphs and visualizations illustrate the impact of parameter changes.
- **Hands-On Labs:** Experiment with different scenarios to build intuition about risk and return.

## License

This project is licensed under a [Creative Commons BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).

## Disclaimer

This tool is intended for educational purposes only. It does not provide investment advice. The data and results are simulated and may not reflect real market behavior.

## Contact

For any questions or feedback, please contact [Luís Simões da Cunha](mailto:your-email@example.com).

---

Happy learning and experimenting with the Fama–French 3-Factor Model!
