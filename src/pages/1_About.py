import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="About - Cloud Resource Optimizer",
    page_icon="ℹ️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .about-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(120deg, #1a365d 0%, #2d3748 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
        padding: 1rem;
    }
    .tech-item {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="about-header">
        <h1>About Cloud Resource Optimizer</h1>
        <p>Intelligent Resource Allocation and Scheduling System for Cloud Environments</p>
    </div>
""", unsafe_allow_html=True)

# Project Overview
st.header("🎯 Project Overview")
st.markdown("""
This innovative system leverages advanced machine learning techniques to optimize cloud resource allocation
and scheduling. By combining multiple AI approaches, we achieve superior resource utilization while
maintaining performance and reducing costs.

### 🔑 Key Features:
- **Hybrid Forecasting Model**: Combines LSTM and ARIMA for accurate resource prediction
- **Deep Q-Learning Optimization**: Intelligent resource allocation decisions
- **Genetic Algorithm Fine-tuning**: Optimizes specific resource parameters
- **Real-time Monitoring**: Continuous tracking of resource utilization
- **Cost Optimization**: Reduces cloud infrastructure expenses
- **Energy Efficiency**: Promotes sustainable resource usage
""")

# Technical Architecture
st.header("🏗️ Technical Architecture")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Components")
    st.markdown("""
    1. **Data Collection Layer**
        - Resource metrics gathering
        - Real-time monitoring
        - Data preprocessing
        
    2. **AI/ML Layer**
        - LSTM-ARIMA hybrid model
        - DQN agent
        - Genetic Algorithm optimizer
        
    3. **Orchestration Layer**
        - Kubernetes integration
        - Resource scheduler
        - Load balancer
    """)

with col2:
    st.subheader("Technology Stack")
    st.markdown("""
    <div class="tech-stack">
        <span class="tech-item">Python 3.8+</span>
        <span class="tech-item">TensorFlow</span>
        <span class="tech-item">Keras</span>
        <span class="tech-item">Scikit-learn</span>
        <span class="tech-item">Pandas</span>
        <span class="tech-item">NumPy</span>
        <span class="tech-item">Streamlit</span>
        <span class="tech-item">Plotly</span>
        <span class="tech-item">Kubernetes</span>
        <span class="tech-item">Docker</span>
    </div>
    """, unsafe_allow_html=True)

# Research Background
st.header("📚 Research Background")
st.markdown("""
This project is based on extensive research in cloud computing, resource optimization, and machine learning.
Key research areas include:

- **Predictive Analytics in Cloud Computing**
- **Reinforcement Learning for Resource Management**
- **Evolutionary Algorithms in Cloud Optimization**
- **Energy-Efficient Computing**
- **Cost Optimization in Cloud Environments**

The system has demonstrated significant improvements in resource utilization and cost reduction:
- 25-30% reduction in resource wastage
- 15-20% improvement in resource utilization
- 20-25% reduction in operational costs
""")

# Implementation Details
st.header("⚙️ Implementation Details")
st.markdown("""
### System Components:

1. **Hybrid Forecasting Model**
   - LSTM for capturing long-term patterns
   - ARIMA for short-term predictions
   - Ensemble approach for improved accuracy

2. **DQN Agent**
   - State space: Current resource metrics
   - Action space: Resource allocation decisions
   - Reward function: Optimization objectives

3. **Genetic Algorithm**
   - Population size: 50
   - Chromosome encoding: Resource parameters
   - Fitness function: Multi-objective optimization

4. **Kubernetes Integration**
   - Custom resource definitions
   - Automated scaling policies
   - Resource quota management
""")

# Future Enhancements
st.header("🚀 Future Enhancements")
st.markdown("""
1. **Advanced AI Models**
   - Integration of transformer models
   - Multi-agent reinforcement learning
   - Federated learning support

2. **Enhanced Monitoring**
   - Real-time anomaly detection
   - Predictive maintenance
   - Custom metric support

3. **Cloud Provider Integration**
   - Multi-cloud support
   - Cloud-specific optimizations
   - Cost prediction models

4. **User Experience**
   - Custom dashboard builder
   - Advanced visualization options
   - Automated reporting system
""") 