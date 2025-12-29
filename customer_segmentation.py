"""
E-commerce Customer Segmentation Analysis
This module provides functions for customer clustering and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """Customer Segmentation using K-Means Clustering"""
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans = None
        self.pca = None
        self.feature_columns = None
        self.categorical_cols = ['Gender', 'Membership Type', 'City', 'Satisfaction Level']
        self.numeric_cols = ['Age', 'Items Purchased', 'Days Since Last Purchase', 
                            'Average Rating', 'Total Spend', 'Discount Applied']
        
    def load_data(self, filepath):
        """Load customer data from CSV"""
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self, data):
        """Clean and prepare data"""
        print("\n=== Data Cleaning ===")
        print(f"Missing values before cleaning:\n{data.isnull().sum()}")
        
        # Remove missing values
        data_clean = data.dropna().copy()
        print(f"\nRows removed: {len(data) - len(data_clean)}")
        print(f"Data shape after cleaning: {data_clean.shape}")
        
        return data_clean
    
    def exploratory_analysis(self, data):
        """Perform exploratory data analysis"""
        print("\n=== Exploratory Data Analysis ===")
        print("\nBasic Statistics:")
        print(data.describe())
        
        # Analysis by Age
        total_spend_age = data.groupby('Age')['Total Spend'].sum()
        print("\nTotal Spend by Age:")
        print(total_spend_age.head())
        
        # Analysis by Membership Type
        membership_spend = data.groupby('Membership Type')['Total Spend'].sum()
        print("\nTotal Spend by Membership Type:")
        print(membership_spend)
        
        # Analysis by City
        city_spend = data.groupby('City')['Total Spend'].sum()
        print(f"\nTotal Spend by City (Top 5):")
        print(city_spend.nlargest(5))
        
        return total_spend_age, membership_spend, city_spend
    
    def visualize_eda(self, data, total_spend_age, membership_spend, city_spend):
        """Create EDA visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total Spend by Age
        axes[0, 0].plot(total_spend_age.index, total_spend_age.values, 
                       marker='o', linestyle='-', color='#3498db')
        axes[0, 0].set_title('Total Spend by Age', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Total Spend')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Total Spend by Membership Type
        colors = ['#e74c3c', '#2ecc71', '#f39c12']
        axes[0, 1].bar(membership_spend.index, membership_spend.values, color=colors)
        axes[0, 1].set_title('Total Spend by Membership Type', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Membership Type')
        axes[0, 1].set_ylabel('Total Spend')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Total Spend by City (Top 10)
        top_cities = city_spend.nlargest(10)
        axes[1, 0].barh(range(len(top_cities)), top_cities.values, color='skyblue')
        axes[1, 0].set_yticks(range(len(top_cities)))
        axes[1, 0].set_yticklabels(top_cities.index)
        axes[1, 0].set_title('Total Spend by City (Top 10)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Total Spend')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Distribution of Total Spend
        axes[1, 1].hist(data['Total Spend'], bins=30, color='#9b59b6', edgecolor='black')
        axes[1, 1].set_title('Distribution of Total Spend', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Total Spend')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def preprocess_data(self, data):
        """Encode and scale features"""
        data_processed = data.copy()
        
        # Label encode categorical columns
        print("\n=== Encoding Categorical Features ===")
        for col in self.categorical_cols:
            if col in data_processed.columns:
                le = LabelEncoder()
                data_processed[col] = le.fit_transform(data_processed[col])
                self.label_encoders[col] = le
                print(f"Encoded: {col}")
        
        # Select features for clustering
        self.feature_columns = [col for col in self.numeric_cols + self.categorical_cols 
                               if col in data_processed.columns]
        
        # Scale features
        print("\n=== Scaling Features ===")
        X = data_processed[self.feature_columns].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Features used: {self.feature_columns}")
        print(f"Scaled data shape: {X_scaled.shape}")
        
        return data_processed, X_scaled
    
    def find_optimal_clusters(self, X_scaled, k_range=(2, 11)):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("\n=== Finding Optimal Clusters ===")
        inertias = []
        silhouette_scores = []
        K_range = range(k_range[0], k_range[1])
        
        for k in K_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            sil_score = silhouette_score(X_scaled, km.labels_)
            silhouette_scores.append(sil_score)
            print(f"k={k}: Inertia={km.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        # Find best k based on silhouette score
        best_k = K_range[np.argmax(silhouette_scores)]
        print(f"\nRecommended k based on silhouette score: {best_k}")
        
        return list(K_range), inertias, silhouette_scores, best_k
    
    def plot_cluster_metrics(self, K_range, inertias, silhouette_scores):
        """Plot elbow curve and silhouette scores"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(K_range, inertias, marker='o', color='#3498db', linewidth=2, markersize=8)
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette scores
        axes[1].plot(K_range, silhouette_scores, marker='o', color='#e74c3c', linewidth=2, markersize=8)
        axes[1].set_title('Silhouette Scores', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Mark best k
        best_k = K_range[np.argmax(silhouette_scores)]
        axes[1].axvline(x=best_k, color='green', linestyle='--', label=f'Best k={best_k}')
        axes[1].legend()
        
        plt.tight_layout()
        return fig
    
    def fit_model(self, X_scaled, n_clusters=None):
        """Fit K-Means model"""
        if n_clusters is not None:
            self.n_clusters = n_clusters
        
        print(f"\n=== Fitting K-Means with {self.n_clusters} clusters ===")
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        self.kmeans.fit(X_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, self.kmeans.labels_)
        print(f"Silhouette Score: {sil_score:.3f}")
        
        return self.kmeans.labels_
    
    def visualize_clusters(self, X_scaled, labels):
        """Visualize clusters using PCA"""
        print("\n=== Creating PCA Visualization ===")
        
        # PCA for visualization
        self.pca = PCA(n_components=3, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        variance_explained = self.pca.explained_variance_ratio_
        print(f"Variance explained by PC1: {variance_explained[0]:.2%}")
        print(f"Variance explained by PC2: {variance_explained[1]:.2%}")
        print(f"Total variance explained: {sum(variance_explained[:2]):.2%}")
        
        # Segment names
        segment_names = {
            0: "High Spenders",
            1: "Loyal Customers",
            2: "Occasional Shoppers",
            3: "Dissatisfied Customers"
        }
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2', 'PCA3'])
        pca_df['Cluster'] = labels
        pca_df['Cluster_Name'] = pca_df['Cluster'].map(
            lambda x: segment_names.get(x, f"Cluster {x}")
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 2D PCA plot
        for cluster in sorted(pca_df['Cluster'].unique()):
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            axes[0].scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
                          label=cluster_data['Cluster_Name'].iloc[0],
                          s=100, alpha=0.6)
        
        axes[0].set_title('Customer Segments - PCA Projection (2D)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'PCA Component 1 ({variance_explained[0]:.1%} variance)', fontsize=12)
        axes[0].set_ylabel(f'PCA Component 2 ({variance_explained[1]:.1%} variance)', fontsize=12)
        axes[0].legend(title='Customer Segment', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 3D visualization data
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(122, projection='3d')
        
        for cluster in sorted(pca_df['Cluster'].unique()):
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], cluster_data['PCA3'],
                      label=cluster_data['Cluster_Name'].iloc[0], s=50, alpha=0.6)
        
        ax.set_title('Customer Segments - PCA Projection (3D)', fontsize=14, fontweight='bold')
        ax.set_xlabel('PCA1', fontsize=10)
        ax.set_ylabel('PCA2', fontsize=10)
        ax.set_zlabel('PCA3', fontsize=10)
        ax.legend(title='Segment', fontsize=8)
        
        plt.tight_layout()
        return fig, pca_df
    
    def create_cluster_profiles(self, data_processed, labels):
        """Create detailed cluster profiles"""
        print("\n=== Creating Cluster Profiles ===")
        
        # Add cluster labels
        data_with_clusters = data_processed.copy()
        data_with_clusters['Cluster'] = labels
        
        # Numeric profiles
        numeric_features = [col for col in self.numeric_cols if col in data_with_clusters.columns]
        cluster_profile_num = data_with_clusters.groupby('Cluster')[numeric_features].mean()
        
        # Categorical profiles (most common value)
        categorical_features = [col for col in self.categorical_cols if col in data_with_clusters.columns]
        cluster_profile_cat = data_with_clusters.groupby('Cluster')[categorical_features].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        
        # Cluster sizes
        cluster_sizes = data_with_clusters['Cluster'].value_counts().sort_index()
        
        print("\n--- Numeric Profile ---")
        print(cluster_profile_num.round(2))
        print("\n--- Categorical Profile ---")
        print(cluster_profile_cat)
        print("\n--- Cluster Sizes ---")
        print(cluster_sizes)
        
        return cluster_profile_num, cluster_profile_cat, cluster_sizes
    
    def visualize_cluster_profiles(self, cluster_profile_num, cluster_sizes):
        """Visualize cluster profiles"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cluster sizes
        axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values, color='#3498db')
        axes[0, 0].set_title('Cluster Sizes', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Average Rating by Cluster
        if 'Average Rating' in cluster_profile_num.columns:
            axes[0, 1].bar(cluster_profile_num.index, cluster_profile_num['Average Rating'], 
                          color='#2ecc71')
            axes[0, 1].set_title('Average Rating by Cluster', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Cluster')
            axes[0, 1].set_ylabel('Average Rating')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Total Spend by Cluster
        if 'Total Spend' in cluster_profile_num.columns:
            axes[1, 0].bar(cluster_profile_num.index, cluster_profile_num['Total Spend'], 
                          color='#e74c3c')
            axes[1, 0].set_title('Average Total Spend by Cluster', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Average Total Spend')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Age by Cluster
        if 'Age' in cluster_profile_num.columns:
            axes[1, 1].bar(cluster_profile_num.index, cluster_profile_num['Age'], 
                          color='#f39c12')
            axes[1, 1].set_title('Average Age by Cluster', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Cluster')
            axes[1, 1].set_ylabel('Average Age')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def predict_cluster(self, customer_data):
        """Predict cluster for new customer data"""
        if self.kmeans is None:
            raise ValueError("Model not trained yet. Please fit the model first.")
        
        # Create DataFrame from input
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        else:
            customer_df = customer_data.copy()
        
        # Encode categorical features with handling for unseen labels
        for col in self.categorical_cols:
            if col in customer_df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                try:
                    # Try to transform the value
                    customer_df[col] = le.transform(customer_df[col])
                except ValueError as e:
                    # Handle unseen labels by using most common label from training
                    print(f"Warning: '{customer_df[col].iloc[0]}' not seen in training for '{col}'. Using most common value.")
                    # Get the most common class (index 0 is typically the first/most common)
                    most_common_encoded = 0  # Default to first encoded value
                    customer_df[col] = most_common_encoded
        
        # Select and order features
        X = customer_df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict cluster
        cluster = self.kmeans.predict(X_scaled)
        
        return cluster[0] if len(cluster) == 1 else cluster
    
    def save_model(self, filepath='customer_segmentation_model.pkl'):
        """Save trained model and preprocessors"""
        model_package = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'pca': self.pca,
            'feature_columns': self.feature_columns,
            'n_clusters': self.n_clusters,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\nModel saved successfully to {filepath}")
    
    def load_model(self, filepath='customer_segmentation_model.pkl'):
        """Load trained model and preprocessors"""
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        self.kmeans = model_package['kmeans']
        self.scaler = model_package['scaler']
        self.label_encoders = model_package['label_encoders']
        self.pca = model_package['pca']
        self.feature_columns = model_package['feature_columns']
        self.n_clusters = model_package['n_clusters']
        self.categorical_cols = model_package['categorical_cols']
        self.numeric_cols = model_package['numeric_cols']
        
        print(f"\nModel loaded successfully from {filepath}")


def main():
    """Main execution function"""
    # Initialize model
    segmenter = CustomerSegmentation(n_clusters=4)
    
    # Load data (update with your file path)
    data = segmenter.load_data('E-commerce Customer Behavior - Sheet1.csv')
    
    if data is None:
        return
    
    # Clean data
    data_clean = segmenter.clean_data(data)
    
    # EDA
    total_spend_age, membership_spend, city_spend = segmenter.exploratory_analysis(data_clean)
    fig_eda = segmenter.visualize_eda(data_clean, total_spend_age, membership_spend, city_spend)
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Preprocess data
    data_processed, X_scaled = segmenter.preprocess_data(data_clean)
    
    # Find optimal clusters
    K_range, inertias, silhouette_scores, best_k = segmenter.find_optimal_clusters(X_scaled)
    fig_metrics = segmenter.plot_cluster_metrics(K_range, inertias, silhouette_scores)
    plt.savefig('cluster_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Fit model with optimal k
    labels = segmenter.fit_model(X_scaled, n_clusters=best_k)
    
    # Visualize clusters
    fig_clusters, pca_df = segmenter.visualize_clusters(X_scaled, labels)
    plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create cluster profiles
    profile_num, profile_cat, sizes = segmenter.create_cluster_profiles(data_processed, labels)
    fig_profiles = segmenter.visualize_cluster_profiles(profile_num, sizes)
    plt.savefig('cluster_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model
    segmenter.save_model('customer_segmentation_model.pkl')
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
