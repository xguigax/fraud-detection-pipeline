"""
Fraud Detection Monitoring Dashboard
Real-time visualization of model performance and predictions
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
from datetime import datetime
import numpy as np

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Fraud Detection Monitoring Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title(" Dashboard Controls")
st.sidebar.markdown("---")

# Load data functions
@st.cache_data(ttl=5)
def load_predictions():
    """Load predictions from database"""
    conn = sqlite3.connect('data/fraud_predictions.db')
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df

@st.cache_data
def load_training_metrics():
    """Load training metrics"""
    try:
        with open('models/training_metrics.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data(ttl=1)
def load_streaming_results():
    """Load streaming simulation results"""
    try:
        return pd.read_csv('data/streaming_results.csv')
    except:
        return None

# Refresh button
if st.sidebar.button(" Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Load all data
predictions_df = load_predictions()
training_metrics = load_training_metrics()
streaming_df = load_streaming_results()

# Sidebar info
st.sidebar.markdown("###  Dataset Info")
if len(predictions_df) > 0:
    st.sidebar.info(f"""
    **Total Predictions:** {len(predictions_df):,}  
    **Fraud Flagged:** {predictions_df['fraud_flag'].sum():,}  
    **Latest Update:** {predictions_df.iloc[0]['timestamp'][:19]}
    """)
else:
    st.sidebar.warning("No predictions yet. Run the streaming simulator!")

# Main dashboard
tabs = st.tabs([" Overview", " Model Performance", " Real-Time Monitoring", " Analysis"])

# ==================== TAB 1: OVERVIEW ====================
with tabs[0]:
    if len(predictions_df) == 0:
        st.warning(" No predictions available. Please run the streaming simulator first.")
    else:
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Transactions",
                value=f"{len(predictions_df):,}",
                delta=None
            )
        
        with col2:
            fraud_count = predictions_df['fraud_flag'].sum()
            fraud_rate = (fraud_count / len(predictions_df)) * 100
            st.metric(
                label="Fraud Detected",
                value=f"{fraud_count:,}",
                delta=f"{fraud_rate:.2f}%"
            )
        
        with col3:
            avg_prob = predictions_df['fraud_probability'].mean()
            st.metric(
                label="Avg Fraud Probability",
                value=f"{avg_prob:.4f}",
                delta=None
            )
        
        with col4:
            high_risk = (predictions_df['fraud_probability'] > 0.7).sum()
            st.metric(
                label="High Risk Transactions",
                value=f"{high_risk:,}",
                delta=f"{(high_risk/len(predictions_df)*100):.2f}%"
            )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fraud Flag Distribution")
            flag_counts = predictions_df['fraud_flag'].value_counts()
            fig = px.pie(
                values=flag_counts.values,
                names=['Normal', 'Fraud'],
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Fraud Probability Distribution")
            fig = px.histogram(
                predictions_df,
                x='fraud_probability',
                nbins=50,
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(
                xaxis_title="Fraud Probability",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent high-risk transactions
        st.subheader(" Recent High-Risk Transactions (Top 10)")
        high_risk_df = predictions_df[predictions_df['fraud_probability'] > 0.7].head(10)
        
        if len(high_risk_df) > 0:
            display_df = high_risk_df[['transaction_id', 'timestamp', 'amount', 'fraud_probability', 'fraud_flag']].copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['fraud_probability'] = display_df['fraud_probability'].round(4)
            display_df['amount'] = display_df['amount'].round(2)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No high-risk transactions detected yet.")

# ==================== TAB 2: MODEL PERFORMANCE ====================
with tabs[1]:
    st.header(" Model Training & Evaluation")
    
    if training_metrics:
        # Model comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("CatBoost Performance")
            cb_metrics = training_metrics['catboost']
            st.metric("ROC-AUC", f"{cb_metrics['roc_auc']:.4f}")
            st.metric("PR-AUC", f"{cb_metrics['pr_auc']:.4f}")
            st.metric("F1-Score", f"{cb_metrics['f1_score']:.4f}")
        
        with col2:
            st.subheader("LightGBM Performance")
            lgbm_metrics = training_metrics['lightgbm']
            st.metric("ROC-AUC", f"{lgbm_metrics['roc_auc']:.4f}")
            st.metric("PR-AUC", f"{lgbm_metrics['pr_auc']:.4f}")
            st.metric("F1-Score", f"{lgbm_metrics['f1_score']:.4f}")
        
        # Best model
        st.success(f"🏆 Best Model in Production: **{training_metrics['best_model']}**")
        
        st.markdown("---")
        
        # Training metrics comparison chart
        st.subheader("Model Comparison")
        
        metrics_comparison = pd.DataFrame({
            'Metric': ['ROC-AUC', 'PR-AUC', 'F1-Score'],
            'CatBoost': [cb_metrics['roc_auc'], cb_metrics['pr_auc'], cb_metrics['f1_score']],
            'LightGBM': [lgbm_metrics['roc_auc'], lgbm_metrics['pr_auc'], lgbm_metrics['f1_score']]
        })
        
        fig = px.bar(
            metrics_comparison,
            x='Metric',
            y=['CatBoost', 'LightGBM'],
            barmode='group',
            color_discrete_sequence=['#3498db', '#e67e22']
        )
        fig.update_layout(yaxis_title="Score", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Training metrics not found. Please run the training script first.")
    


# ==================== TAB 3: REAL-TIME MONITORING ====================
with tabs[2]:
    st.header(" Real-Time Monitoring")
    
    if len(predictions_df) == 0:
        st.warning("No predictions to monitor yet.")
    else:
        # Time-series analysis
        predictions_df['timestamp_dt'] = pd.to_datetime(predictions_df['timestamp'])
        predictions_df = predictions_df.sort_values('timestamp_dt')
        
        # Add batch numbers for grouping
        predictions_df['batch'] = (predictions_df.index // 50)
        
        # Aggregate by batch
        batch_stats = predictions_df.groupby('batch').agg({
            'fraud_probability': 'mean',
            'fraud_flag': 'sum',
            'transaction_id': 'count',
            'amount': 'mean'
        }).reset_index()
        batch_stats.columns = ['batch', 'avg_fraud_prob', 'fraud_count', 'transaction_count', 'avg_amount']
        batch_stats['fraud_rate'] = (batch_stats['fraud_count'] / batch_stats['transaction_count']) * 100
        
        # Fraud probability over time
        st.subheader("Average Fraud Probability Over Time")
        fig = px.line(
            batch_stats,
            x='batch',
            y='avg_fraud_prob',
            markers=True,
            color_discrete_sequence=['#e74c3c']
        )
        fig.update_layout(
            xaxis_title="Batch (50 transactions each)",
            yaxis_title="Average Fraud Probability"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud rate over time
        st.subheader("Fraud Detection Rate Over Time")
        fig = px.bar(
            batch_stats,
            x='batch',
            y='fraud_rate',
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(
            xaxis_title="Batch (50 transactions each)",
            yaxis_title="Fraud Rate (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average transaction amount over time
        st.subheader("Average Transaction Amount Over Time")
        fig = px.line(
            batch_stats,
            x='batch',
            y='avg_amount',
            markers=True,
            color_discrete_sequence=['#2ecc71']
        )
        fig.update_layout(
            xaxis_title="Batch (50 transactions each)",
            yaxis_title="Average Amount ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 4: ANALYSIS ====================
with tabs[3]:
    st.header(" Detailed Analysis")
    
    if len(predictions_df) == 0:
        st.warning("No data to analyze yet.")
    else:
        # Amount vs Fraud Probability
        st.subheader("Transaction Amount vs Fraud Probability")
        
        sample_size = min(1000, len(predictions_df))
        sample_df = predictions_df.sample(n=sample_size, random_state=42)
        
        fig = px.scatter(
            sample_df,
            x='amount',
            y='fraud_probability',
            color='fraud_flag',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={'fraud_flag': 'Fraud Flag'},
            opacity=0.6
        )
        fig.update_layout(
            xaxis_title="Transaction Amount ($)",
            yaxis_title="Fraud Probability"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud probability distribution by amount range
        st.subheader("Fraud Risk by Transaction Amount Range")
        
        predictions_df['amount_range'] = pd.cut(
            predictions_df['amount'],
            bins=[0, 50, 100, 200, 500, 1000, 10000],
            labels=['$0-50', '$50-100', '$100-200', '$200-500', '$500-1K', '$1K+']
        )
        
        amount_analysis = predictions_df.groupby('amount_range').agg({
            'fraud_probability': 'mean',
            'fraud_flag': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        amount_analysis.columns = ['amount_range', 'avg_fraud_prob', 'fraud_count', 'total_count']
        amount_analysis['fraud_rate'] = (amount_analysis['fraud_count'] / amount_analysis['total_count']) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                amount_analysis,
                x='amount_range',
                y='avg_fraud_prob',
                color_discrete_sequence=['#9b59b6']
            )
            fig.update_layout(
                xaxis_title="Amount Range",
                yaxis_title="Average Fraud Probability"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                amount_analysis,
                x='amount_range',
                y='fraud_rate',
                color_discrete_sequence=['#e67e22']
            )
            fig.update_layout(
                xaxis_title="Amount Range",
                yaxis_title="Fraud Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader(" Raw Predictions Data")
        st.dataframe(
            predictions_df[['transaction_id', 'timestamp', 'amount', 'fraud_probability', 'fraud_flag']].head(100),
            use_container_width=True,
            hide_index=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p> Fraud Detection Dashboard | Built with Streamlit & Plotly</p>
    <p>Last Updated: {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)