# 🛍️ E-commerce Customer Segmentation

A comprehensive customer segmentation solution using K-Means clustering with an interactive Streamlit dashboard for analysis and real-time predictions.

## 📋 Features

### 🎯 Core Functionality
- **Data Loading & Cleaning**: Handle missing values and prepare data for analysis
- **Exploratory Data Analysis**: Interactive visualizations and statistical summaries
- **K-Means Clustering**: Automatic customer segmentation with optimal cluster detection
- **PCA Visualization**: 2D and 3D cluster visualizations
- **Real-time Prediction**: Predict customer segments for new customers
- **Model Persistence**: Save and load trained models

### 📊 Dashboard Features
- **Data Overview**: View dataset statistics and missing data information
- **Interactive EDA**: Explore distributions, correlations, and categorical analyses
- **Cluster Analysis**: Visualize segments with interactive 3D plots
- **Prediction Interface**: Input customer data and get instant segment predictions
- **Actionable Insights**: Receive marketing recommendations for each segment

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Install required packages:**
```bash
pip install -r requirements.txt
```

2. **Prepare your data:**
   - Use the sample data generator (built-in), OR
   - Prepare your CSV file with the following columns:
     - Customer ID
     - Age
     - Gender
     - City
     - Membership Type
     - Total Spend
     - Items Purchased
     - Average Rating
     - Days Since Last Purchase
     - Satisfaction Level
     - Discount Applied

### Running the Application

#### Option 1: Streamlit Dashboard (Recommended)
```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### Option 2: Python Script
```bash
python customer_segmentation.py
```

Make sure to update the file path in `customer_segmentation.py`:
```python
data = segmenter.load_data('your_data_file.csv')
```

## 📖 Usage Guide

### 1. Load Data
- **Option A**: Upload your CSV file using the sidebar
- **Option B**: Click "Generate Sample Data" to create demo data

### 2. Configure Model
- Select number of clusters (2-10)
- Enable "Auto-find optimal k" for automatic cluster selection
- Click "Train Model"

### 3. Explore Results
- **Data Overview Tab**: View dataset statistics and preview
- **EDA Analysis Tab**: Explore feature distributions and correlations
- **Clustering Results Tab**: Visualize segments and profiles
- **Prediction Tab**: Predict segments for new customers

### 4. Make Predictions
- Enter customer information in the Prediction tab
- Click "Predict Segment"
- View segment assignment and recommendations

## 🎨 Customer Segments

The model identifies 4 main customer segments:

### 💎 High Spenders (Cluster 0)
- **Characteristics**: High total spend, premium membership
- **Strategy**: VIP treatment, exclusive offers, premium products

### ⭐ Loyal Customers (Cluster 1)
- **Characteristics**: Regular purchases, high satisfaction
- **Strategy**: Loyalty rewards, regular engagement, personalized offers

### 🛒 Occasional Shoppers (Cluster 2)
- **Characteristics**: Infrequent purchases, price-sensitive
- **Strategy**: Targeted promotions, time-limited discounts

### ⚠️ Dissatisfied Customers (Cluster 3)
- **Characteristics**: Low satisfaction, declining engagement
- **Strategy**: Follow-up, compensation offers, service improvement

## 📊 Model Performance

The system uses multiple metrics to ensure quality clustering:

- **Elbow Method**: Identifies the optimal number of clusters
- **Silhouette Score**: Measures cluster separation quality
- **PCA Analysis**: Reduces dimensionality for visualization

## 🔧 Advanced Features

### Save Trained Model
```python
from customer_segmentation import CustomerSegmentation

model = CustomerSegmentation(n_clusters=4)
# ... train model ...
model.save_model('my_model.pkl')
```

### Load Existing Model
```python
model = CustomerSegmentation()
model.load_model('my_model.pkl')

# Make predictions
cluster = model.predict_cluster({
    'Age': 35,
    'Gender': 'Male',
    'Membership Type': 'Gold',
    # ... other features
})
```

## 📈 Visualizations

The application provides:
- Distribution plots for all numeric features
- Categorical analysis with spending patterns
- Correlation heatmaps
- 2D and 3D PCA cluster visualizations
- Interactive Plotly charts
- Cluster comparison dashboards

## 🛠️ Technical Details

### Technologies Used
- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive visualizations
- **Matplotlib/Seaborn**: Static visualizations

### Algorithms
- **K-Means Clustering**: Customer segmentation
- **PCA**: Dimensionality reduction for visualization
- **StandardScaler**: Feature normalization
- **LabelEncoder**: Categorical feature encoding

## 📝 Code Improvements Made

### From Original Code:
1. ✅ Fixed typos (e.g., "membershipp_type_total_spend")
2. ✅ Added all missing imports
3. ✅ Organized code into classes and functions
4. ✅ Implemented proper error handling
5. ✅ Added model persistence (save/load)
6. ✅ Created comprehensive EDA functions
7. ✅ Implemented automatic optimal k detection
8. ✅ Added prediction functionality
9. ✅ Created interactive visualizations
10. ✅ Built production-ready Streamlit app
11. ✅ Added sample data generator
12. ✅ Implemented cluster profiling
13. ✅ Added marketing recommendations

## 🎯 Future Enhancements

- [ ] Add more clustering algorithms (DBSCAN, Hierarchical)
- [ ] Implement customer lifetime value prediction
- [ ] Add export functionality for reports
- [ ] Include A/B testing framework
- [ ] Add real-time data streaming support
- [ ] Implement user authentication
- [ ] Add database integration
- [ ] Create API endpoints

## 📞 Support

For issues or questions:
1. Check the console for error messages
2. Ensure all dependencies are installed
3. Verify your data format matches the requirements
4. Review the sample data structure

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- Built with Streamlit for interactive dashboards
- Uses Scikit-learn for machine learning
- Plotly for interactive visualizations

---

**Happy Segmenting! 🎉**
