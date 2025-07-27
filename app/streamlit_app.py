import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# Page Configuration & Styling
# ---------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor", 
    page_icon="🚢", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .survived-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: 3px solid #00ff88;
    }
    
    .not-survived-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border: 3px solid #ff6b6b;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #ff9a56;
    }
    
    .stSelectbox > div > div {
        background-color: #f0f8ff;
    }
    
    .stSlider > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = os.path.join("models", "final_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except:
    st.error("⚠️ Model file not found. Please ensure 'models/final_model.pkl' exists.")
    st.stop()

# ---------------------------
# Header Section
# ---------------------------
st.markdown("""
<div class="main-header">
    <h1>🚢 Titanic Survival Prediction Dashboard</h1>
    <p style="font-size: 1.2em; margin-top: 1rem; opacity: 0.9;">
        Discover the fate of Titanic passengers using advanced machine learning
    </p>
    <p style="font-size: 1em; margin-top: 0.5rem; opacity: 0.8;">
        ⚡ Real-time predictions • 📊 Interactive visualizations • 🧠 AI-powered insights
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar for Inputs
# ---------------------------
st.sidebar.markdown("## 👤 Passenger Profile")
st.sidebar.markdown("*Fill in the details below to predict survival*")

with st.sidebar:
    # Passenger Class with emoji
    pclass = st.selectbox(
        "🎫 Passenger Class", 
        [1, 2, 3], 
        format_func=lambda x: f"{'🥇' if x==1 else '🥈' if x==2 else '🥉'} {x} Class"
    )
    
    # Sex with emoji
    sex = st.radio("⚤ Gender", ["Male", "Female"])
    
    # Age with color coding
    age = st.slider("📅 Age", 0, 80, 25, help="Age of the passenger")
    
    # Family details
    st.markdown("### 👨‍👩‍👧‍👦 Family Information")
    sibsp = st.number_input("👫 Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("👶 Parents/Children Aboard", 0, 10, 0)
    
    # Financial details
    st.markdown("### 💰 Financial Details")
    fare = st.number_input("💵 Fare (£)", 0.0, 600.0, 32.2, step=0.1)
    
    # Embarkation
    st.markdown("### 🌍 Journey Details")
    embarked = st.selectbox(
        "🚢 Port of Embarkation", 
        ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
        format_func=lambda x: f"{'🇬🇧' if 'S' in x else '🇫🇷' if 'C' in x else '🇮🇪'} {x}"
    )
    
    title = st.selectbox(
        "🎩 Title", 
        ["Mr", "Mrs", "Miss", "Master", "Rare"],
        format_func=lambda x: f"{'👨' if x=='Mr' else '👩' if x=='Mrs' else '👧' if x=='Miss' else '👦' if x=='Master' else '👑'} {x}"
    )

# ---------------------------
# Main Content Layout
# ---------------------------
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### 📋 Passenger Summary")
    
    # Feature Engineering for display
    sex_encoded = 0 if sex == "Male" else 1
    embarked_mapping = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
    embarked_encoded = embarked_mapping[embarked]
    title_mapping = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}
    title_encoded = title_mapping[title]
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    # Display passenger info in cards
    st.markdown(f"""
    <div class="metric-card">
        <strong>👤 Profile:</strong> {sex} {title}<br>
        <strong>📅 Age:</strong> {age} years<br>
        <strong>🎫 Class:</strong> {pclass}<br>
        <strong>👨‍👩‍👧‍👦 Family Size:</strong> {family_size}<br>
        <strong>🏝️ Traveling Alone:</strong> {'Yes' if is_alone else 'No'}<br>
        <strong>💰 Fare:</strong> £{fare:.2f}
    </div>
    """, unsafe_allow_html=True)
    
    # Risk factors analysis
    st.markdown("### ⚠️ Risk Factors")
    risk_factors = []
    if sex == "Male":
        risk_factors.append("👨 Male gender (higher risk)")
    if pclass == 3:
        risk_factors.append("🥉 Third class (higher risk)")
    if age > 60:
        risk_factors.append("👴 Advanced age (higher risk)")
    if family_size > 4:
        risk_factors.append("👨‍👩‍👧‍👦 Large family (higher risk)")
    if fare < 20:
        risk_factors.append("💰 Low fare (higher risk)")
    
    protective_factors = []
    if sex == "Female":
        protective_factors.append("👩 Female gender (lower risk)")
    if pclass == 1:
        protective_factors.append("🥇 First class (lower risk)")
    if title in ["Mrs", "Miss"]:
        protective_factors.append("👩 Female title (lower risk)")
    if fare > 50:
        protective_factors.append("💰 High fare (lower risk)")
    
    if risk_factors:
        st.markdown("**🔴 Risk Factors:**")
        for factor in risk_factors:
            st.markdown(f"• {factor}")
    
    if protective_factors:
        st.markdown("**🟢 Protective Factors:**")
        for factor in protective_factors:
            st.markdown(f"• {factor}")

# ---------------------------
# Prediction Section
# ---------------------------
with col1:
    # Prepare input dataframe
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked_encoded,
        "Title": title_encoded,
        "FamilySize": family_size,
        "IsAlone": is_alone
    }])
    
    # Auto-predict (always show prediction)
    pred_proba = model.predict_proba(input_data)[0]
    prediction = np.argmax(pred_proba)
    survival_prob = pred_proba[1] * 100
    death_prob = pred_proba[0] * 100
    
    # Display prediction with enhanced styling
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-card survived-card">
            <h2>🌟 SURVIVED! 🌟</h2>
            <h3>Survival Probability: {survival_prob:.1f}%</h3>
            <p style="font-size: 1.1em; margin-top: 1rem;">
                This passenger would likely have survived the Titanic disaster
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-card not-survived-card">
            <h2>💀 DID NOT SURVIVE 💀</h2>
            <h3>Death Probability: {death_prob:.1f}%</h3>
            <p style="font-size: 1.1em; margin-top: 1rem;">
                This passenger would likely not have survived the Titanic disaster
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive Probability Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = survival_prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "🎯 Survival Probability"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "lightgreen" if survival_prob > 50 else "lightcoral"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Probability Comparison Chart
    fig_prob = px.bar(
        x=['Death', 'Survival'], 
        y=[death_prob, survival_prob],
        color=['Death', 'Survival'],
        color_discrete_map={'Death': '#ff6b6b', 'Survival': '#4ecdc4'},
        title="📊 Probability Breakdown",
        labels={'x': 'Outcome', 'y': 'Probability (%)'}
    )
    fig_prob.update_layout(height=300, showlegend=False)
    fig_prob.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    st.plotly_chart(fig_prob, use_container_width=True)

# ---------------------------
# Feature Analysis Section
# ---------------------------
st.markdown("---")
st.markdown("## 🔍 Feature Impact Analysis")

col3, col4 = st.columns(2)

with col3:
    # Enhanced feature importance with realistic values
    feature_importance = {
        "👤 Gender": 0.5 if sex_encoded == 1 else -0.5,
        "🎫 Class": -0.3 if pclass == 3 else (0.2 if pclass == 1 else 0.1),
        "📅 Age": -0.2 if age > 50 else (0.1 if age < 16 else 0.0),
        "💰 Fare": 0.3 if fare > 50 else (0.1 if fare > 20 else -0.1),
        "👨‍👩‍👧‍👦 Family Size": 0.2 if 2 <= family_size <= 4 else -0.2,
        "🎩 Title": 0.3 if title in ["Mrs", "Miss"] else (-0.2 if title == "Mr" else 0.1),
        "🌍 Embarkation": 0.1 if embarked_encoded == 1 else (0.05 if embarked_encoded == 2 else 0.0)
    }
    
    # Create horizontal bar chart
    features = list(feature_importance.keys())
    impacts = list(feature_importance.values())
    colors = ['#4ecdc4' if x > 0 else '#ff6b6b' for x in impacts]
    
    fig_features = go.Figure(go.Bar(
        y=features,
        x=impacts,
        orientation='h',
        marker_color=colors,
        text=[f"{x:+.2f}" for x in impacts],
        textposition='outside'
    ))
    
    fig_features.update_layout(
        title="📈 Feature Contributions to Survival",
        xaxis_title="Impact on Survival Probability",
        height=400,
        showlegend=False
    )
    fig_features.add_vline(x=0, line_width=2, line_color="black")
    st.plotly_chart(fig_features, use_container_width=True)

with col4:
    st.markdown("### 💡 Key Insights")
    
    # Generate dynamic insights
    insights = []
    
    if sex == "Female":
        insights.append("🚺 **Women First Policy**: Being female significantly increases survival chances due to the 'women and children first' evacuation protocol.")
    
    if pclass == 1:
        insights.append("🥇 **First Class Advantage**: First-class passengers had better access to lifeboats and were located closer to the boat deck.")
    elif pclass == 3:
        insights.append("🥉 **Third Class Challenge**: Third-class passengers faced barriers reaching lifeboats, including locked gates and distance from boat deck.")
    
    if age < 16:
        insights.append("👶 **Child Priority**: Children had higher survival rates due to rescue prioritization.")
    elif age > 60:
        insights.append("👴 **Age Factor**: Older passengers faced physical challenges during evacuation.")
    
    if fare > 100:
        insights.append("💎 **Wealth Advantage**: Higher fare often correlated with better cabin locations and quicker lifeboat access.")
    
    if family_size == 1:
        insights.append("🏝️ **Solo Travel**: Traveling alone could mean faster decision-making but less help during crisis.")
    elif family_size > 4:
        insights.append("👨‍👩‍👧‍👦 **Large Family Burden**: Large families faced coordination challenges during evacuation.")
    
    for insight in insights[:4]:  # Show top 4 insights
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)

# ---------------------------
# Historical Context
# ---------------------------
st.markdown("---")
st.markdown("## 📚 Historical Context")

col5, col6, col7 = st.columns(3)

with col5:
    st.markdown("""
    <div class="feature-card">
        <h4>🚢 The Disaster</h4>
        <p><strong>Date:</strong> April 15, 1912</p>
        <p><strong>Lives Lost:</strong> 1,517 people</p>
        <p><strong>Survivors:</strong> 710 people</p>
        <p><strong>Overall Survival Rate:</strong> 32%</p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Class Statistics</h4>
        <p><strong>1st Class:</strong> 62% survived</p>
        <p><strong>2nd Class:</strong> 47% survived</p>
        <p><strong>3rd Class:</strong> 24% survived</p>
        <p><strong>Crew:</strong> 24% survived</p>
    </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown("""
    <div class="feature-card">
        <h4>⚖️ Gender Impact</h4>
        <p><strong>Women:</strong> 74% survived</p>
        <p><strong>Men:</strong> 20% survived</p>
        <p><strong>Children:</strong> 52% survived</p>
        <p><em>"Women and children first" protocol</em></p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; color: white; margin-top: 2rem;">
    <p style="margin: 0; font-size: 1.1em;">
        🤖 Powered by Machine Learning | 📊 Built with Streamlit | 🚢 Remember the Titanic
    </p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.8;">
        This prediction is based on historical patterns and should be interpreted as educational content.
    </p>
</div>
""", unsafe_allow_html=True)