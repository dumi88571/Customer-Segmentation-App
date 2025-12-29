"""
E-commerce Customer Segmentation - Streamlit App
Interactive dashboard for customer clustering and prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from customer_segmentation import CustomerSegmentation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 20px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e 0%, #283593 100%);
    }
    h1 {
        color: #ffffff;
        text-align: center;
        font-weight: 800;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    h2 {
        color: #e3f2fd;
        border-bottom: 3px solid #00bcd4;
        padding-bottom: 10px;
    }
    h3 {
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00bcd4 0%, #0097a7 100%);
        color: white;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 188, 212, 0.6);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    /* Fix text colors for dark theme */
    .stMarkdown, p, div, span, label {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

def load_sample_data():
    """Generate sample e-commerce customer data"""
    np.random.seed(42)
    n_samples = 500
    
    genders = ['Male', 'Female']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
              'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    membership_types = ['Gold', 'Silver', 'Bronze']
    satisfaction_levels = ['Satisfied', 'Neutral', 'Unsatisfied']
    
    data = {
        'Customer ID': [f'CUST{i:04d}' for i in range(1, n_samples + 1)],
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(genders, n_samples),
        'City': np.random.choice(cities, n_samples),
        'Membership Type': np.random.choice(membership_types, n_samples, p=[0.2, 0.5, 0.3]),
        'Total Spend': np.random.gamma(2, 200, n_samples).round(2),
        'Items Purchased': np.random.randint(1, 50, n_samples),
        'Average Rating': np.random.uniform(1, 5, n_samples).round(2),
        'Days Since Last Purchase': np.random.randint(1, 365, n_samples),
        'Satisfaction Level': np.random.choice(satisfaction_levels, n_samples, p=[0.6, 0.3, 0.1]),
        'Discount Applied': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    return pd.DataFrame(data)

def plot_interactive_clusters(pca_df):
    """Create interactive 3D cluster visualization"""
    fig = px.scatter_3d(
        pca_df, 
        x='PCA1', 
        y='PCA2', 
        z='PCA3',
        color='Cluster_Name',
        title='Customer Segments - Interactive 3D View',
        labels={'PCA1': 'Component 1', 'PCA2': 'Component 2', 'PCA3': 'Component 3'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(
        height=600,
        font=dict(size=12),
        title_font_size=18,
        scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            zaxis_title='PCA Component 3'
        )
    )
    
    return fig

def plot_cluster_comparison(profile_num, cluster_sizes):
    """Create interactive cluster comparison charts"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cluster Sizes', 'Average Rating by Cluster', 
                       'Total Spend by Cluster', 'Age Distribution by Cluster'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Cluster sizes
    fig.add_trace(
        go.Bar(x=cluster_sizes.index, y=cluster_sizes.values, 
               name='Size', marker_color='#3498db'),
        row=1, col=1
    )
    
    # Average Rating
    if 'Average Rating' in profile_num.columns:
        fig.add_trace(
            go.Bar(x=profile_num.index, y=profile_num['Average Rating'], 
                   name='Rating', marker_color='#2ecc71'),
            row=1, col=2
        )
    
    # Total Spend
    if 'Total Spend' in profile_num.columns:
        fig.add_trace(
            go.Bar(x=profile_num.index, y=profile_num['Total Spend'], 
                   name='Spend', marker_color='#e74c3c'),
            row=2, col=1
        )
    
    # Age
    if 'Age' in profile_num.columns:
        fig.add_trace(
            go.Bar(x=profile_num.index, y=profile_num['Age'], 
                   name='Age', marker_color='#f39c12'),
            row=2, col=2
        )
    
    fig.update_layout(height=700, showlegend=False, title_text="Cluster Analysis Dashboard")
    
    return fig

def map_cluster_to_segment(cluster_id, profile_num):
    """Dynamically map cluster ID to segment type based on actual cluster characteristics"""
    try:
        if cluster_id not in profile_num.index:
            return {"name": f"Cluster {cluster_id}", "type": "unknown", "color": "#95a5a6", "icon": "📊"}
        
        cluster_data = profile_num.loc[cluster_id]
        avg_spend = cluster_data.get('Total Spend', 0) if hasattr(cluster_data, 'get') else 0
        avg_rating = cluster_data.get('Average Rating', 0) if hasattr(cluster_data, 'get') else 0
        avg_days_since = cluster_data.get('Days Since Last Purchase', 180) if hasattr(cluster_data, 'get') else 180
        
        # Calculate percentiles across ALL clusters for better classification
        all_spends = []
        all_ratings = []
        
        if 'Total Spend' in profile_num.columns:
            all_spends = [x for x in profile_num['Total Spend'].values if x is not None and not pd.isna(x)]
        if 'Average Rating' in profile_num.columns:
            all_ratings = [x for x in profile_num['Average Rating'].values if x is not None and not pd.isna(x)]
        
        # Safety checks
        if len(all_spends) == 0:
            all_spends = [0]
        if len(all_ratings) == 0:
            all_ratings = [0]
        
        spend_percentile = (avg_spend >= sorted(all_spends)[-1] * 0.85) if len(all_spends) > 0 else False
        high_rating = avg_rating >= 4.0
        low_rating = avg_rating < 3.0
        median_spend = sorted(all_spends)[len(all_spends)//2] if len(all_spends) > 0 else 0
        low_spend = avg_spend < median_spend
        
        # Rank-based classification with stricter criteria
        if spend_percentile and avg_rating >= 3.8:
            # Only the TOP spending cluster with good ratings
            return {"name": "High Spenders 💎", "type": "high_spender", "color": "#FFD700", "icon": "💎"}
        elif low_rating or (avg_rating < 3.2 and avg_spend < 300):
            # Low satisfaction or low engagement
            return {"name": "Dissatisfied Customers ⚠️", "type": "dissatisfied", "color": "#E74C3C", "icon": "⚠️"}
        elif low_spend or avg_days_since > 60:
            # Below median spend or infrequent shoppers
            return {"name": "Occasional Shoppers 🛒", "type": "occasional", "color": "#F39C12", "icon": "🛒"}
        else:
            # Default: consistent moderate spenders
            return {"name": "Loyal Customers ⭐", "type": "loyal", "color": "#2ECC71", "icon": "⭐"}
    
    except Exception as e:
        # Fallback in case of any error
        return {"name": f"Cluster {cluster_id}", "type": "unknown", "color": "#95a5a6", "icon": "📊"}


# Main App
st.title("🛍️ E-commerce Customer Segmentation Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source:",
        ["Upload CSV", "Use Sample Data"]
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Upload your customer data CSV file"
        )
        
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.info(f"Shape: {st.session_state.data.shape}")
    else:
        if st.button("🎲 Generate Sample Data"):
            st.session_state.data = load_sample_data()
            st.success("✅ Sample data generated!")
            st.info(f"Shape: {st.session_state.data.shape}")
    
    st.markdown("---")
    
    # Model configuration
    st.header("🔧 Model Settings")
    n_clusters = st.number_input(
        "Number of Clusters", 
        min_value=2, 
        max_value=10, 
        value=4,
        step=1,
        help="Select the number of customer segments",
        key="n_clusters_input"
    )
    
    auto_find_k = st.checkbox(
        "Auto-find optimal k", 
        value=False,
        help="Automatically determine the best number of clusters"
    )
    
    st.markdown("---")
    
    # Train model button
    if st.button("🚀 Train Model", use_container_width=True):
        if st.session_state.data is not None:
            with st.spinner("Training model... Please wait..."):
                try:
                    # Initialize model
                    st.session_state.model = CustomerSegmentation(n_clusters=n_clusters)
                    
                    # Clean data
                    data_clean = st.session_state.model.clean_data(st.session_state.data)
                    
                    # Preprocess
                    data_processed, X_scaled = st.session_state.model.preprocess_data(data_clean)
                    
                    # Find optimal k if requested
                    if auto_find_k:
                        K_range, inertias, sil_scores, best_k = st.session_state.model.find_optimal_clusters(X_scaled)
                        n_clusters = best_k
                        st.info(f"Optimal k found: {best_k}")
                    
                    # Fit model
                    labels = st.session_state.model.fit_model(X_scaled, n_clusters=n_clusters)
                    
                    # Store results
                    st.session_state.data_clean = data_clean
                    st.session_state.data_processed = data_processed
                    st.session_state.X_scaled = X_scaled
                    st.session_state.labels = labels
                    st.session_state.trained = True
                    
                    # Save model
                    st.session_state.model.save_model('customer_segmentation_model.pkl')
                    
                    st.success("✅ Model trained successfully!")
                except Exception as e:
                    st.error(f"❌ Error training model: {e}")
        else:
            st.warning("⚠️ Please load data first!")

# Main content area
if st.session_state.data is not None:
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview", 
        "🔍 EDA Analysis", 
        "🎯 Clustering Results", 
        "🔮 Prediction"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(st.session_state.data):,}")
        
        with col2:
            st.metric("Features", len(st.session_state.data.columns))
        
        with col3:
            missing_pct = (st.session_state.data.isnull().sum().sum() / 
                          (len(st.session_state.data) * len(st.session_state.data.columns)) * 100)
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        with col4:
            if 'Total Spend' in st.session_state.data.columns:
                avg_spend = st.session_state.data['Total Spend'].mean()
                st.metric("Avg. Spend", f"${avg_spend:.2f}")
        
        st.markdown("---")
        
        # Display data
        st.subheader("📋 Dataset Preview")
        st.dataframe(st.session_state.data.head(20), use_container_width=True)
        
        # Statistics
        st.subheader("📈 Statistical Summary")
        st.dataframe(st.session_state.data.describe(), use_container_width=True)
    
    # Tab 2: EDA Analysis
    with tab2:
        st.header("🔍 Exploratory Data Analysis")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Distribution plots
            st.subheader("📊 Feature Distributions")
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_feature = st.selectbox("Select feature to visualize:", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(
                        data, 
                        x=selected_feature,
                        title=f'Distribution of {selected_feature}',
                        color_discrete_sequence=['#3498db']
                    )
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(
                        data, 
                        y=selected_feature,
                        title=f'Box Plot of {selected_feature}',
                        color_discrete_sequence=['#e74c3c']
                    )
                    fig_box.update_layout(height=400)
                    st.plotly_chart(fig_box, use_container_width=True)
            
            st.markdown("---")
            
            # Categorical analysis
            st.subheader("📊 Categorical Analysis")
            
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols and 'Total Spend' in data.columns:
                selected_cat = st.selectbox("Select categorical feature:", categorical_cols)
                
                # Group by category
                cat_analysis = data.groupby(selected_cat)['Total Spend'].agg(['sum', 'mean', 'count'])
                cat_analysis = cat_analysis.sort_values('sum', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_bar = px.bar(
                        cat_analysis.reset_index(),
                        x=selected_cat,
                        y='sum',
                        title=f'Total Spend by {selected_cat}',
                        color='sum',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    fig_avg = px.bar(
                        cat_analysis.reset_index(),
                        x=selected_cat,
                        y='mean',
                        title=f'Average Spend by {selected_cat}',
                        color='mean',
                        color_continuous_scale='Plasma'
                    )
                    st.plotly_chart(fig_avg, use_container_width=True)
            
            # Correlation heatmap
            st.markdown("---")
            st.subheader("🔥 Correlation Heatmap")
            
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title='Feature Correlations'
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
    
    # Tab 3: Clustering Results
    with tab3:
        st.header("🎯 Clustering Results")
        
        if st.session_state.trained:
            model = st.session_state.model
            labels = st.session_state.labels
            X_scaled = st.session_state.X_scaled
            data_processed = st.session_state.data_processed
            
            # Cluster visualization
            st.subheader("🌟 Cluster Visualization")
            
            with st.spinner("Creating visualizations..."):
                # PCA visualization
                fig_pca, pca_df = model.visualize_clusters(X_scaled, labels)
                st.pyplot(fig_pca)
                
                # Interactive 3D plot
                st.subheader("🎨 Interactive 3D Cluster View")
                fig_3d = plot_interactive_clusters(pca_df)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            st.markdown("---")
            
            # Cluster profiles
            st.subheader("📋 Cluster Profiles")
            
            profile_num, profile_cat, cluster_sizes = model.create_cluster_profiles(
                data_processed, labels
            )
            
            # Display profiles
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numeric Features Profile:**")
                st.dataframe(profile_num.round(2), use_container_width=True)
            
            with col2:
                st.write("**Categorical Features Profile:**")
                st.dataframe(profile_cat, use_container_width=True)
            
            # Cluster sizes
            st.markdown("---")
            st.subheader("📊 Cluster Distribution")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    cluster_sizes.reset_index().rename(
                        columns={'index': 'Cluster', 0: 'Count'}
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                fig_pie = px.pie(
                    values=cluster_sizes.values,
                    names=[f'Cluster {i}' for i in cluster_sizes.index],
                    title='Customer Distribution Across Segments',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Interactive comparison
            st.markdown("---")
            st.subheader("📊 Cluster Comparison Dashboard")
            fig_comparison = plot_cluster_comparison(profile_num, cluster_sizes)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
        else:
            st.info("👈 Please train the model first using the sidebar.")
    
    # Tab 4: Prediction
    with tab4:
        st.header("🔮 Customer Segment Prediction")
        
        if st.session_state.trained:
            model = st.session_state.model
            
            st.markdown("""
            Enter customer information below to predict which segment they belong to.
            This helps in targeted marketing and personalized customer experience.
            """)
            
            st.markdown("---")
            
            # Input form
            st.subheader("📝 Customer Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30, key="pred_age")
                gender = st.selectbox("Gender", ["Male", "Female"], key="pred_gender")
                membership = st.selectbox("Membership Type", ["Gold", "Silver", "Bronze"], key="pred_membership")
            
            with col2:
                total_spend = st.number_input("Total Spend ($)", min_value=0.0, value=500.0, step=10.0, key="pred_spend")
                items = st.number_input("Items Purchased", min_value=1, max_value=100, value=10, key="pred_items")
                rating = st.slider("Average Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1, key="pred_rating")
            
            with col3:
                city = st.selectbox("City", [
                    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
                ], key="pred_city")
                days_since = st.number_input("Days Since Last Purchase", min_value=1, max_value=365, value=30, key="pred_days")
                satisfaction = st.selectbox("Satisfaction Level", ["Satisfied", "Neutral", "Unsatisfied"], key="pred_satisfaction")
                discount = st.selectbox("Discount Applied", [0, 1], key="pred_discount")
            
            st.markdown("---")
            
            # Predict button
            if st.button("🎯 Predict Segment", use_container_width=True):
                
                # Create customer data dictionary
                customer_data = {
                    'Age': age,
                    'Gender': gender,
                    'Membership Type': membership,
                    'Total Spend': total_spend,
                    'Items Purchased': items,
                    'Average Rating': rating,
                    'City': city,
                    'Days Since Last Purchase': days_since,
                    'Satisfaction Level': satisfaction,
                    'Discount Applied': discount
                }
                
                try:
                    # Predict cluster
                    cluster = model.predict_cluster(customer_data)
                    
                    # Get cluster profiles for better segment determination
                    profile_num, profile_cat, _ = model.create_cluster_profiles(
                        st.session_state.data_processed, 
                        st.session_state.labels
                    )
                    
                    # Dynamically map cluster to segment type based on actual characteristics
                    segment_info = map_cluster_to_segment(cluster, profile_num)
                    segment_name = segment_info["name"]
                    segment_type = segment_info["type"]
                    segment_color = segment_info["color"]
                    segment_icon = segment_info["icon"]
                    
                    # Display prediction with custom color
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {segment_color}22 0%, {segment_color}44 100%); 
                                padding: 20px; border-radius: 10px; border-left: 5px solid {segment_color}; 
                                text-align: center; margin: 20px 0;">
                        <h2 style="color: {segment_color}; margin: 0;">🎯 Predicted Segment: {segment_name}</h2>
                        <p style="color: #ffffff; margin-top: 10px;">ML Cluster: {cluster} | Based on spending, satisfaction, and engagement patterns</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Two column layout for details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"📊 Customer Profile Analysis")
                        st.write("**Input Characteristics:**")
                        st.metric("Total Spend", f"${total_spend:.2f}")
                        st.metric("Items Purchased", items)
                        st.metric("Average Rating", f"{rating:.1f}/5.0")
                        st.metric("Days Since Last Purchase", days_since)
                        st.metric("Membership Level", membership)
                        st.metric("Satisfaction", satisfaction)
                        
                        st.markdown("---")
                        st.write("**📊 Why This Segment?**")
                        cluster_stats = profile_num.loc[cluster]
                        st.info(f"""
                        **Cluster {cluster} Average Metrics:**
                        - Avg Spend: ${cluster_stats.get('Total Spend', 0):.2f}
                        - Avg Rating: {cluster_stats.get('Average Rating', 0):.2f}/5.0
                        - Avg Days Since Purchase: {cluster_stats.get('Days Since Last Purchase', 0):.0f} days
                        
                        This customer matches the behavior pattern of the **{segment_name}** segment.
                        """)
                    
                    with col2:
                        st.subheader("💡 Comprehensive Marketing Recommendations")
                        
                        # HIGH SPENDERS
                        if segment_type == "high_spender":
                            recommendations = [
                                "🎁 **VIP Program Enrollment**: Enroll in exclusive VIP membership with concierge service",
                                "💎 **Premium Product Showcase**: Send curated selection of high-end products",
                                "🎉 **Exclusive Event Invitations**: Invite to private shopping events and product launches",
                                "🎯 **Personalized Shopping**: Assign dedicated personal shopper",
                                "📦 **White Glove Delivery**: Offer premium delivery and installation services",
                                "💳 **Elite Credit Line**: Provide special financing options",
                                "🏆 **First Access**: Early access to new collections and limited editions",
                                "🎨 **Custom Orders**: Ability to request custom or personalized items",
                                "🌟 **Loyalty Multiplier**: 3x points on all purchases",
                                "🍾 **Complimentary Gifts**: Include luxury gift with next purchase",
                                "📞 **Priority Support**: 24/7 dedicated customer service hotline",
                                "✈️ **Travel Perks**: Partner benefits (hotels, airlines, restaurants)"
                            ]
                        
                        # LOYAL CUSTOMERS
                        elif segment_type == "loyal":
                            recommendations = [
                                "⭐ **Loyalty Rewards Program**: Earn points on every purchase (2x current rate)",
                                "🎂 **Birthday Rewards**: Special discount on birthday month",
                                "📧 **Personalized Emails**: Weekly curated product recommendations",
                                "🎁 **Referral Bonuses**: Earn $50 credit for each friend referred",
                                "💰 **Member-Only Sales**: Access to exclusive flash sales",
                                "📱 **Mobile App Benefits**: Extra 10% off when ordering via app",
                                "🏅 **Tier Upgrade**: Fast-track to next membership level",
                                "💌 **Thank You Notes**: Personalized appreciation messages",
                                "🎊 **Anniversary Gifts**: Celebrate customer anniversary with special offers",
                                "📰 **Newsletter Exclusives**: First to know about new products",
                                "🤝 **Community Access**: Join customer advisory board",
                                "🎯 **Smart Recommendations**: AI-powered personalized suggestions"
                            ]
                        
                        # OCCASIONAL SHOPPERS
                        elif segment_type == "occasional":
                            recommendations = [
                                "⚡ **Time-Limited Flash Sales**: 48-hour exclusive discounts up to 40% off",
                                "🎁 **Welcome Back Offer**: Special 25% discount on next purchase",
                                "📦 **Free Shipping**: Complimentary shipping on orders over $50",
                                "🔔 **Price Drop Alerts**: Notifications when wishlist items go on sale",
                                "💳 **Easy Payment Plans**: Buy now, pay later options available",
                                "🛍️ **Bundle Deals**: Save 30% when buying product bundles",
                                "📧 **Abandoned Cart Recovery**: 15% off items left in cart",
                                "🎉 **Seasonal Promotions**: Holiday and seasonal sale announcements",
                                "💰 **First-Time Buyer Discount**: Additional savings for category newcomers",
                                "📱 **SMS Deal Alerts**: Instant text notifications for exclusive mobile deals",
                                "🎯 **Targeted Promotions**: Deals based on browsing history",
                                "🏷️ **Clearance Access**: Early notification of clearance events"
                            ]
                        
                        # DISSATISFIED CUSTOMERS
                        else:  # dissatisfied
                            recommendations = [
                                "🆘 **Immediate Follow-Up**: Priority customer service call within 24 hours",
                                "💬 **Personal Apology**: Direct message from customer care manager",
                                "🎁 **Compensation Offer**: $100 credit toward next purchase",
                                "📋 **Feedback Survey**: Earn $25 credit for completing satisfaction survey",
                                "🔧 **Issue Resolution**: Dedicated team to resolve any outstanding problems",
                                "💸 **Hassle-Free Returns**: Extended 90-day return policy",
                                "🎯 **Service Recovery**: Complimentary product or service upgrade",
                                "📞 **Direct Line**: Personal extension to customer retention specialist",
                                "✨ **Quality Guarantee**: 100% satisfaction guarantee on next order",
                                "🏆 **Win-Back Offer**: Exclusive 50% discount to regain trust",
                                "📦 **Free Premium Shipping**: Complimentary expedited delivery",
                                "🤝 **Relationship Rebuild**: Monthly check-ins to ensure satisfaction",
                                "⚠️ **Priority Queue**: Skip the line for all future support needs"
                            ]
                        
                        # Display recommendations
                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"{i}. {rec}")
                        
                        st.markdown("---")
                        st.success(f"✅ **Total Recommendations**: {len(recommendations)} actionable strategies")
                    
                    # Additional insights section
                    st.markdown("---")
                    st.subheader("📈 Behavioral Insights & Next Steps")
                    
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        st.write("**💰 Revenue Potential**")
                        if segment_type == "high_spender":
                            st.success("🔥 **Very High** - Top 20% of customers")
                            potential = "Focus on retention and lifetime value maximization"
                        elif segment_type == "loyal":
                            st.info("📈 **High** - Consistent revenue contributor")
                            potential = "Opportunity to upgrade to premium tier"
                        elif segment_type == "occasional":
                            st.warning("📊 **Medium** - Untapped potential")
                            potential = "Increase purchase frequency through engagement"
                        else:
                            st.error("⚠️ **At Risk** - Potential churn")
                            potential = "Critical: Prevent customer loss"
                        st.caption(potential)
                    
                    with col4:
                        st.write("**🎯 Engagement Strategy**")
                        if segment_type == "high_spender":
                            strategy = "White-glove service"
                            st.success(f"💎 {strategy}")
                            st.caption("Maintain premium experience")
                        elif segment_type == "loyal":
                            strategy = "Reward & nurture"
                            st.info(f"⭐ {strategy}")
                            st.caption("Build long-term relationship")
                        elif segment_type == "occasional":
                            strategy = "Re-engagement campaign"
                            st.warning(f"🔔 {strategy}")
                            st.caption("Increase touchpoints")
                        else:
                            strategy = "Service recovery"
                            st.error(f"🆘 {strategy}")
                            st.caption("Urgent intervention required")
                    
                    with col5:
                        st.write("**📊 Communication Channel**")
                        if segment_type == "high_spender":
                            st.success("📞 Personal calls + Email")
                            st.caption("High-touch, personalized")
                        elif segment_type == "loyal":
                            st.info("📧 Email + Push notifications")
                            st.caption("Regular, engaging content")
                        elif segment_type == "occasional":
                            st.warning("📱 SMS + Social media")
                            st.caption("Attention-grabbing offers")
                        else:
                            st.error("☎️ Direct phone outreach")
                            st.caption("Personal intervention")
                    
                    # NEW SECTION: Show all 4 clusters detailed information
                    st.markdown("---")
                    st.markdown("---")
                    st.header("📊 All Customer Segments Overview")
                    st.markdown("""
                    Here's a complete breakdown of all customer segments identified by the clustering model.
                    Understanding these segments helps in crafting targeted strategies for each group.
                    """)
                    
                    # Pre-calculate all segments to handle duplicates with descriptive names
                    cluster_segments = {}
                    segment_groups = {}
                    
                    # First pass: Group clusters by segment name
                    for i in range(model.n_clusters):
                        seg = map_cluster_to_segment(i, profile_num)
                        name = seg['name']
                        if name not in segment_groups:
                            segment_groups[name] = []
                        segment_groups[name].append(i)
                        cluster_segments[i] = seg
                    
                    # Second pass: Differentiate duplicates
                    for name, cluster_ids in segment_groups.items():
                        if len(cluster_ids) > 1:
                            # If duplicates exist, compare them to give meaningful suffixes
                            # We'll compare based on Total Spend
                            spends = {cid: profile_num.loc[cid, 'Total Spend'] for cid in cluster_ids}
                            sorted_cids = sorted(cluster_ids, key=lambda x: spends[x])
                            
                            # Assign suffixes based on relative spend
                            for rank, cid in enumerate(sorted_cids):
                                seg = cluster_segments[cid]
                                if len(cluster_ids) == 2:
                                    suffix = " (Lower Spend)" if rank == 0 else " (Higher Spend)"
                                else:
                                    suffix = f" (Tier {rank + 1})"
                                seg['name'] = f"{seg['name'].split(' ')[0]} {seg['name'].split(' ')[1]} {suffix} {seg['icon']}"
                    
                    # Display each cluster in a card
                    for cluster_id in range(model.n_clusters):
                        segment = cluster_segments[cluster_id]
                        
                        # Get cluster statistics
                        cluster_stats = profile_num.loc[cluster_id] if cluster_id in profile_num.index else None
                        
                        with st.expander(f"{segment['icon']} **{segment['name']}** (Cluster {cluster_id})", expanded=(cluster_id == cluster)):
                            col_a, col_b = st.columns([1, 2])
                            
                            with col_a:
                                st.markdown(f"### {segment['icon']} Profile Statistics")
                                if cluster_stats is not None:
                                    st.metric("Avg Total Spend", f"${cluster_stats.get('Total Spend', 0):.2f}")
                                    st.metric("Avg Age", f"{cluster_stats.get('Age', 0):.0f} years")
                                    st.metric("Avg Rating", f"{cluster_stats.get('Average Rating', 0):.2f}/5.0")
                                    st.metric("Avg Items", f"{cluster_stats.get('Items Purchased', 0):.0f}")
                                    st.metric("Avg Days Since Purchase", f"{cluster_stats.get('Days Since Last Purchase', 0):.0f}")
                                
                                # Show if this is their predicted cluster
                                if cluster_id == cluster:
                                    st.success("✅ **YOUR PREDICTED SEGMENT**")
                            
                            with col_b:
                                st.markdown(f"### 💡 Key Characteristics & Recommendations")
                                
                                # Use segment type to determine content
                                if segment['type'] == "high_spender":
                                    st.markdown("""
                                    **Characteristics:**
                                    - High spending patterns (typically >$1000)
                                    - Excellent satisfaction ratings (4.0+)
                                    - Premium membership preference
                                    - High engagement and purchase frequency
                                    
                                    **Top Recommendations:**
                                    1. 🎁 VIP Program Enrollment - Exclusive concierge service
                                    2. 💎 Premium Product Showcase - Curated high-end selections
                                    3. 🎉 Private Event Invitations - Product launches & exclusive events
                                    4. 🏆 Early Access - New collections and limited editions
                                    5. ✈️ Travel Perks - Partner benefits (hotels, airlines)
                                    """)
                                
                                elif segment['type'] == "loyal":  # Loyal Customers
                                    st.markdown("""
                                    **Characteristics:**
                                    - Consistent purchasing behavior
                                    - Good satisfaction levels (3.5-4.5)
                                    - Moderate to good spending ($300-$1000)
                                    - Regular engagement with brand
                                    
                                    **Top Recommendations:**
                                    1. ⭐ Loyalty Rewards - 2x points on all purchases
                                    2. 🎂 Birthday Rewards - Special birthday month discounts
                                    3. 🎁 Referral Bonuses - $50 credit per friend referred
                                    4. 📱 Mobile App Benefits - Extra 10% off via app
                                    5. 🤝 Community Access - Join customer advisory board
                                    """)
                                
                                elif segment['type'] == "occasional":  # Occasional Shoppers
                                    st.markdown("""
                                    **Characteristics:**
                                    - Infrequent purchases (90+ days between orders)
                                    - Lower spending levels (<$300)
                                    - Mixed satisfaction ratings
                                    - Price-sensitive behavior
                                    
                                    **Top Recommendations:**
                                    1. ⚡ Flash Sales - 48-hour exclusive discounts (40% off)
                                    2. 🎁 Welcome Back Offer - 25% discount on next purchase
                                    3. 📦 Free Shipping - Complimentary shipping over $50
                                    4. 💳 Easy Payment Plans - Buy now, pay later options
                                    5. 🛍️ Bundle Deals - Save 30% on product bundles
                                    """)
                                
                                elif segment['type'] == "dissatisfied":  # Dissatisfied
                                    st.markdown("""
                                    **Characteristics:**
                                    - Low satisfaction ratings (<3.0)
                                    - At-risk of churning
                                    - Recent negative experiences
                                    - Requires immediate attention
                                    
                                    **Top Recommendations:**
                                    1. 🆘 Immediate Follow-Up - Priority call within 24 hours
                                    2. 💬 Personal Apology - Direct message from manager
                                    3. 🎁 Compensation Offer - $100 credit on next purchase
                                    4. 🏆 Win-Back Offer - Exclusive 50% discount
                                    5. 🤝 Relationship Rebuild - Monthly satisfaction check-ins
                                    """)
                                else:
                                    st.info("Custom segment - analyze patterns to determine best approach")
                
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    st.exception(e)
        
        else:
            st.info("👈 Please train the model first using the sidebar.")

else:
    st.info("👈 Please load data from the sidebar to get started!")
    
    # Welcome message
    st.markdown("""
    ## Welcome to the Customer Segmentation Dashboard! 🎉
    
    This application helps you:
    - 📊 **Analyze** customer behavior patterns
    - 🎯 **Segment** customers using K-Means clustering
    - 🔮 **Predict** which segment new customers belong to
    - 💡 **Gain insights** for targeted marketing
    
    ### Getting Started:
    1. Load your customer data or use sample data (sidebar)
    2. Configure model settings
    3. Train the clustering model
    4. Explore results and make predictions!
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>E-commerce Customer Segmentation Dashboard | Built with Streamlit & Scikit-learn</p>
    </div>
""", unsafe_allow_html=True)
