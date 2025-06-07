import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from standalone import StandaloneOptimizer
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Cloud Resource Optimizer",
    page_icon="☁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = StandaloneOptimizer()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Header
st.title("☁️ Intelligent Cloud Resource Optimizer")
st.markdown("""
This application uses advanced machine learning techniques to optimize cloud resource allocation:
- **Hybrid Forecasting** (LSTM + ARIMA)
- **Deep Q-Learning** for resource decisions
- **Genetic Algorithm** for fine-tuning
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool helps optimize cloud resource allocation using:
    - 🧠 AI-Powered Predictions
    - 📊 Real-time Analysis
    - 💰 Cost Optimization
    - ⚡ Energy Efficiency
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Upload your CSV dataset
    2. Wait for the model training
    3. View optimization results
    4. Monitor metrics in real-time
    """)

# Main content
tab1, tab2 = st.tabs(["📈 Resource Optimization", "📊 Data Analysis"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload your resource metrics CSV file",
            type=['csv'],
            help="File should contain: timestamp, cpu_utilization, memory_utilization, running_pods, total_pods"
        )
        
        if uploaded_file:
            try:
                # Read and process data
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                
                # Train models
                with st.spinner("Training models... This may take a few moments."):
                    st.session_state.optimizer.train_models(df)
                
                # Get optimization results
                results = st.session_state.optimizer.optimize_resources()
                st.session_state.results = results
                
                # Success message
                st.success("✅ Models trained successfully!")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with col2:
        st.header("Current Status")
        if st.session_state.results:
            with st.container():
                st.metric("CPU Request", st.session_state.results['cpu_request'])
                st.metric("Memory Request", st.session_state.results['memory_request'])
                st.metric("Replicas", st.session_state.results['replicas'])
                st.metric("Predicted Utilization", 
                         f"{st.session_state.results['predicted_utilization']:.2%}")

    # Visualization section
    if st.session_state.data is not None:
        st.header("Resource Utilization Trends")
        col3, col4 = st.columns(2)
        
        with col3:
            # CPU and Memory Utilization
            fig_util = go.Figure()
            fig_util.add_trace(go.Scatter(
                x=st.session_state.data['timestamp'],
                y=st.session_state.data['cpu_utilization'] * 100,
                name='CPU Utilization',
                line=dict(color='#1f77b4')
            ))
            fig_util.add_trace(go.Scatter(
                x=st.session_state.data['timestamp'],
                y=st.session_state.data['memory_utilization'] * 100,
                name='Memory Utilization',
                line=dict(color='#2ca02c')
            ))
            fig_util.update_layout(
                title='CPU and Memory Utilization Over Time',
                xaxis_title='Time',
                yaxis_title='Utilization (%)',
                height=400
            )
            st.plotly_chart(fig_util, use_container_width=True)
            
        with col4:
            # Pod Distribution
            latest_data = st.session_state.data.iloc[-1]
            running_pods = latest_data['running_pods']
            available_pods = latest_data['total_pods'] - running_pods
            
            fig_pods = go.Figure(data=[go.Pie(
                labels=['Running Pods', 'Available Pods'],
                values=[running_pods, available_pods],
                hole=.3,
                marker_colors=['#9467bd', '#d3d3d3']
            )])
            fig_pods.update_layout(
                title='Pod Distribution',
                height=400
            )
            st.plotly_chart(fig_pods, use_container_width=True)

with tab2:
    if st.session_state.data is not None:
        st.header("Detailed Metrics Analysis")
        
        # Time series analysis
        st.subheader("Resource Usage Patterns")
        fig_patterns = px.line(st.session_state.data, 
                             x='timestamp', 
                             y=['cpu_utilization', 'memory_utilization'],
                             title='Resource Usage Over Time')
        st.plotly_chart(fig_patterns, use_container_width=True)
        
        # Statistics
        st.subheader("Statistical Summary")
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("##### Resource Utilization Stats")
            stats_df = st.session_state.data[['cpu_utilization', 'memory_utilization']].describe()
            stats_df = stats_df * 100  # Convert to percentages
            st.dataframe(stats_df.round(2))
            
        with col6:
            st.markdown("##### Pod Distribution Stats")
            pod_stats = st.session_state.data[['running_pods', 'total_pods']].describe()
            st.dataframe(pod_stats.round(2))
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        corr_matrix = st.session_state.data[['cpu_utilization', 'memory_utilization', 'running_pods']].corr()
        fig_corr = px.imshow(corr_matrix,
                            labels=dict(color="Correlation"),
                            color_continuous_scale="RdBu")
        st.plotly_chart(fig_corr, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Cloud Resource Optimizer Team") 