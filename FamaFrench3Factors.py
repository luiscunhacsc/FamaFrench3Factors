import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------- Utility Functions ------------------------------
# These functions set up the default values for the parameters used in the model.
# Each parameter represents a key concept:
#   - market_beta: Sensitivity of the asset to market movements.
#   - smb_beta: Sensitivity to the size factor (SMB, Small Minus Big).
#   - hml_beta: Sensitivity to the value factor (HML, High Minus Low).
#   - alpha: The extra return (or "skill") not explained by market factors.
#   - risk_free: The return of a risk-free asset (like a government bond).
#   - noise: The randomness in the asset’s return (idiosyncratic risk).

def reset_parameters():
    st.session_state["market_beta"] = 1.0
    st.session_state["smb_beta"] = 0.2
    st.session_state["hml_beta"] = -0.3
    st.session_state["alpha"] = 0.005
    st.session_state["risk_free"] = 0.02
    st.session_state["noise"] = 0.02

# Lab 1: Detecting Manager Skill
# Here we want to see if a small extra return (alpha) can be detected when there is little random noise.
def set_lab1_parameters():
    st.session_state["market_beta"] = 1.0
    st.session_state["smb_beta"] = 0.0    # No extra sensitivity to size: neutral position
    st.session_state["hml_beta"] = 0.0     # No extra sensitivity to value: neutral position
    st.session_state["alpha"] = 0.005      # A small extra return of 0.5% per month (alpha)
    st.session_state["risk_free"] = 0.02
    st.session_state["noise"] = 0.01       # Very little random variation (noise)

# Lab 2: Factor Timing Strategy
# In this lab we simulate a situation where the strategy actively tilts toward small companies (SMB)
# when they are expected to outperform. We set a higher SMB beta.
def set_lab2_parameters():
    st.session_state["market_beta"] = 1.2
    st.session_state["smb_beta"] = 1.0    # High sensitivity to the size factor (favoring small caps)
    st.session_state["hml_beta"] = 1.1
    st.session_state["alpha"] = -0.003     # A small negative alpha here
    st.session_state["risk_free"] = 0.03
    st.session_state["noise"] = 0.03

# Lab 3: Crisis Period Analysis
# This lab simulates a market crisis scenario:
#   - A lower market sensitivity (market_beta)
#   - Negative exposure to size (SMB) and a strong negative tilt to value (HML)
def set_lab3_parameters():
    st.session_state["market_beta"] = 0.8
    st.session_state["smb_beta"] = -0.7
    st.session_state["hml_beta"] = -1.2
    st.session_state["alpha"] = 0.010
    st.session_state["risk_free"] = 0.01
    st.session_state["noise"] = 0.05

# ---------------------------- Core Model -------------------------------------
# This function generates simulated factor data.
# We create three factors:
#   - Mkt-RF: Market excess return (Market Return minus Risk-Free Rate)
#   - SMB: Size factor (returns of small companies minus large companies)
#   - HML: Value factor (returns of high book-to-market stocks minus low book-to-market stocks)
def generate_ff_data(months=60):
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=months, freq='ME')
    factors = pd.DataFrame({
        'Mkt-RF': np.random.normal(0.05/12, 0.15/np.sqrt(12), months),
        'SMB': np.random.normal(0.03/12, 0.10/np.sqrt(12), months),
        'HML': np.random.normal(0.04/12, 0.12/np.sqrt(12), months)
    }, index=dates)
    return factors

# This function applies the Fama–French 3-Factor model.
# The equation is:
#
#   (Asset Return) - (Risk-Free Rate) = alpha + (market_beta)*(Mkt-RF) 
#             + (smb_beta)*(SMB) + (hml_beta)*(HML) + random noise
#
# Here, each term is:
#   - alpha: Extra return not explained by the factors.
#   - market_beta, smb_beta, hml_beta: The sensitivities (or loadings) to each factor.
#   - Noise: Randomness in returns.
def fama_french_model(params, factors):
    returns = (
        params['alpha'] + 
        params['beta_mkt'] * factors['Mkt-RF'] +
        params['beta_smb'] * factors['SMB'] +
        params['beta_hml'] * factors['HML'] +
        np.random.normal(0, params['noise'], len(factors))
    )
    return returns

# This function runs a regression (a statistical analysis) to see how well our model explains the simulated asset returns.
# It subtracts the risk-free rate from the asset return and uses the factors (with an added constant) as explanatory variables.
def run_regression(stock_returns, factors, rf):
    y = stock_returns - rf
    X = sm.add_constant(factors[['Mkt-RF', 'SMB', 'HML']])
    model = sm.OLS(y, X).fit()
    return model

# ---------------------------- Streamlit App ----------------------------------
st.set_page_config(layout="wide")
st.title("📈 Fama–French 3-Factor Model Playground")
st.markdown(
    """
    This tool helps you understand the Fama–French 3-Factor Model in a simple and interactive way.
    
    **What is the Fama–French Model?**  
    It is a method to explain the returns of a stock by using three key factors:
    
    - **Market Factor (Mkt-RF):** The excess return of the overall market (Market Return minus the Risk-Free Rate).
    - **Size Factor (SMB):** The difference in returns between small and large companies.
    - **Value Factor (HML):** The difference in returns between high and low book-to-market stocks.
    
    The model also includes **alpha (α)**, which is the extra return that cannot be explained by these factors.
    """
)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Model Parameters")
    st.button("↺ Reset Defaults", on_click=reset_parameters)
    
    st.subheader("Factor Loadings")
    st.slider("Market Beta (Sensitivity to market returns, β₁)", -0.5, 2.0, 1.0, key="market_beta")
    st.slider("SMB Beta (Sensitivity to size factor, β₂)", -1.0, 1.0, 0.2, key="smb_beta")
    st.slider("HML Beta (Sensitivity to value factor, β₃)", -1.0, 1.0, -0.3, key="hml_beta")
    
    st.subheader("Other Parameters")
    st.slider("Alpha (Extra return, α)", -0.02, 0.02, 0.005, step=0.001, key="alpha")
    st.slider("Risk-Free Rate (Return of a risk-free asset)", 0.0, 0.05, 0.02, step=0.001, key="risk_free")
    st.slider("Idiosyncratic Volatility (Randomness)", 0.0, 0.05, 0.02, key="noise")

    # Disclaimer and Authorship
    st.markdown("---")
    st.markdown(
    """
    **⚠️ Disclaimer**  
    *For educational purposes only. Not investment advice.*  
    
    <small>
    This tool demonstrates key academic concepts from asset pricing. It uses simulated data to help you understand how different factors affect stock returns.
    </small>
    """, unsafe_allow_html=True)
    
    st.markdown(
    """
    <div style="margin-top: 20px;">
        <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en" target="_blank">
            <img src="https://licensebuttons.net/l/by-nc/4.0/88x31.png" alt="CC BY-NC 4.0">
        </a>
        <br>
        <span style="font-size: 0.8em;">By Luís Simões da Cunha</span>
    </div>
    """, unsafe_allow_html=True)

# Generate simulated factor data and run our model
factors = generate_ff_data()
params = {
    'beta_mkt': st.session_state["market_beta"],
    'beta_smb': st.session_state["smb_beta"],
    'beta_hml': st.session_state["hml_beta"],
    'alpha': st.session_state["alpha"],
    'noise': st.session_state["noise"]
}
stock_returns = fama_french_model(params, factors)
model = run_regression(stock_returns, factors, st.session_state["risk_free"])

# Create tabs for different sections of our app
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Interactive Analysis", 
    "📚 Theoretical Foundation", 
    "📖 Step-by-Step Guide", 
    "🔬 Empirical Labs",
    "🧠 Factor Investing Basics"
])

# ---------------------------- Tab 1: Interactive Analysis ----------------------------
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.success("### Regression Results")
        st.write(f"**Alpha (α):** {model.params['const']:.4f} (p-value: {model.pvalues['const']:.3f})")
        st.write(f"**Market Beta (β₁):** {model.params['Mkt-RF']:.3f} (t-stat: {model.tvalues['Mkt-RF']:.2f})")
        st.write(f"**SMB Beta (β₂):** {model.params['SMB']:.3f} (t-stat: {model.tvalues['SMB']:.2f})")
        st.write(f"**HML Beta (β₃):** {model.params['HML']:.3f} (t-stat: {model.tvalues['HML']:.2f})")
        st.write(f"**R-squared:** {model.rsquared:.3f}")
        
        st.markdown(
        """
        ### Key Metrics Explained
        - **Alpha (α):** Extra return not explained by the market, size, or value factors.
        - **Betas (β):** Measure how much the asset's return moves with each factor.
        - **t-stat:** A value used to determine if the effect of a factor is statistically significant (a value greater than about 2 is often considered significant).
        """)
    
    with col2:
        st.subheader("Factor Exposure Analysis")
        fig, ax = plt.subplots(figsize=(10, 4))
        # Plot the estimated betas for each factor.
        model.params[1:].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title("Estimated Factor Sensitivities", fontweight='bold')
        # Show 95% confidence intervals for the beta estimates.
        ax.errorbar(range(3), model.params[1:], yerr=1.96*model.bse[1:], fmt='none', ecolor='red', capsize=5)
        st.pyplot(fig)
        
        st.subheader("Actual vs. Predicted Returns")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(model.predict(), stock_returns - st.session_state["risk_free"], alpha=0.5)
        ax.plot([-0.2, 0.2], [-0.2, 0.2], 'r--')
        ax.set_xlabel("Model Predicted Excess Returns")
        ax.set_ylabel("Actual Excess Returns (Return - Risk-Free Rate)")
        st.pyplot(fig)

# ---------------------------- Tab 2: Theoretical Foundation ----------------------------
with tab2:
    st.markdown(
    """
    ## Understanding the Fama–French 3-Factor Model
    
    **Model Equation:**  
    The model can be written as:
    
    $$
    (Return) - (Risk-Free) = \\alpha + \\beta_{Mkt}(Mkt-RF) + \\beta_{SMB}(SMB) + \\beta_{HML}(HML) + \\epsilon
    $$
    
    **Explanation of Symbols:**  
    - **Return:** The observed return of the asset.  
    - **Risk-Free:** The return on a risk-free asset (e.g., a government bond).  
    - **α (Alpha):** The extra return (or “manager skill”) not explained by the market factors.  
    - **β (Beta):** The sensitivity of the asset’s return to a particular factor. For example:  
       - **β₍Mkt₎:** How much the asset's return moves with the overall market return.  
       - **β₍SMB₎:** How sensitive the asset is to the size effect (small vs. large companies).  
       - **β₍HML₎:** How sensitive the asset is to the value effect (high vs. low book-to-market stocks).  
    - **Mkt-RF:** The excess return of the market (market return minus risk-free rate).  
    - **SMB:** “Small Minus Big” – the return difference between small and large companies.  
    - **HML:** “High Minus Low” – the return difference between value stocks (high book-to-market) and growth stocks (low book-to-market).  
    - **ε:** The random error term capturing any variation not explained by the factors.
    """)
    
# ---------------------------- Tab 3: Step-by-Step Guide ----------------------------
with tab3:
    st.markdown(
    """
    ## A Simple Guide to the Model and Analysis
    
    **Step 1: Know the Factors**  
    - **Market Factor (Mkt-RF):** Measures how the overall market affects asset returns.  
    - **Size Factor (SMB):** Compares small companies to large companies.  
    - **Value Factor (HML):** Compares companies with high vs. low book-to-market ratios.
    
    **Step 2: Understand the Model Equation**  
    The model tells us that:
    - An asset’s excess return (Return minus Risk-Free) is partly explained by its exposure (beta) to each factor.
    - **Alpha (α)** is the extra return that the model cannot explain.
    
    **Step 3: Run and Interpret the Regression**  
    - We estimate the betas and alpha using regression.
    - **Key Metrics:**  
       - A high beta means a strong relationship with that factor.
       - The t-statistic helps us decide if that relationship is statistically significant.
    
    **Step 4: Visualize the Results**  
    - Compare the predicted returns from the model with the actual returns.
    - Look at the bar charts to see how sensitive your asset is to each factor.
    """)
    
# ---------------------------- Tab 4: Empirical Labs ----------------------------
with tab4:
    st.header("🔍 Empirical Labs")
    lab = st.radio("Choose an Empirical Lab:", [
        "Lab 1: Detecting Manager Skill",
        "Lab 2: Factor Timing Strategy",
        "Lab 3: Crisis Period Analysis"
    ])
    
    if lab == "Lab 1: Detecting Manager Skill":
        st.markdown(
        """
        ### Lab 1: Can We Detect Manager Skill?  
        **Idea:** Even if a manager adds a small extra return (alpha), it might be hard to detect if there's a lot of randomness.  
        
        **Setup:**  
        - **Alpha (Extra Return):** Set to 0.5% per month (0.005)  
        - **Noise:** Very low (1% per month or 0.01), so randomness is minimized  
        
        **What to Observe:**  
        - Run the simulation multiple times and see if the extra return (alpha) shows up as statistically significant.
        """)
        st.button("⚡ Set Lab 1 Parameters", on_click=set_lab1_parameters)
    
    elif lab == "Lab 2: Factor Timing Strategy":
        st.markdown(
        """
        ### Lab 2: Exploring Factor Timing  
        **Idea:** Check if adjusting your portfolio’s exposure can improve returns when market conditions change.
        
        **Setup:**  
        - **SMB Beta:** Set to a high positive value (1.0) to emphasize small company returns when they are expected to do well.  
        - Other parameters are set to simulate a timing strategy.
        
        **What to Observe:**  
        - Compare the model’s performance when the strategy tilts toward small companies.
        """)
        st.button("⚡ Set Lab 2 Parameters", on_click=set_lab2_parameters)
    
    else:
        st.markdown(
        """
        ### Lab 3: Crisis Period Analysis  
        **Idea:** Simulate a crisis scenario where market conditions are unusual.
        
        **Setup:**  
        - **Market Beta:** Reduced to 0.8, meaning the asset is less sensitive to overall market movements.
        - **SMB Beta:** Negative (-0.7), indicating a disadvantage for small companies.
        - **HML Beta:** Strongly negative (-1.2), showing a strong tilt away from value stocks.
        
        **What to Observe:**  
        - See how the asset’s return characteristics change during a simulated market downturn.
        """)
        st.button("⚡ Set Lab 3 Parameters", on_click=set_lab3_parameters)

# ---------------------------- Tab 5: Factor Investing Basics ----------------------------
with tab5:
    st.markdown(
    """
    ## Factor Investing Fundamentals
    
    **What are Factors?**  
    Factors are common sources of risk and return in financial markets. They help explain why different assets perform differently.
    
    **Key Factors in this Model:**  
    - **Market:** Overall market movements.
    - **Size (SMB):** The difference in returns between small and large companies.
    - **Value (HML):** The difference in returns between value and growth stocks.
    
    **Real-World Data (Typical Values):**
    
    | Factor | Annual Premium | Typical Volatility |
    |--------|----------------|--------------------|
    | Market | ~6%            | 15%                |
    | SMB    | ~2%            | 10%                |
    | HML    | ~3%            | 12%                |
    
    **Challenges in Factor Investing:**  
    1. Crowded trades may reduce expected premiums.  
    2. High turnover can lead to increased transaction costs.  
    3. Distinguishing between risk-based and behavioral explanations can be complex.
    
    **Modern Extensions:**  
    Researchers have expanded on this model by adding factors like profitability and investment, and even using machine learning techniques.
    """)
    
st.sidebar.markdown("---")
st.sidebar.markdown(
"""
**Educational Purpose Only**  
*Factor returns are simulated and hypothetical.*  
<small>Data: Simulated Fama–French factors based on historical properties.</small>
""", unsafe_allow_html=True)
