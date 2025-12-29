"""
Example script demonstrating how to use the Customer Segmentation model
for predictions outside of the Streamlit app
"""

from customer_segmentation import CustomerSegmentation
import pandas as pd

def example_1_train_and_predict():
    """Example 1: Train model and make predictions"""
    print("=" * 70)
    print("EXAMPLE 1: Train Model and Make Predictions")
    print("=" * 70)
    
    # Initialize segmenter
    segmenter = CustomerSegmentation(n_clusters=4)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    import numpy as np
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'Customer ID': [f'CUST{i:04d}' for i in range(1, n_samples + 1)],
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago'], n_samples),
        'Membership Type': np.random.choice(['Gold', 'Silver', 'Bronze'], n_samples),
        'Total Spend': np.random.gamma(2, 200, n_samples).round(2),
        'Items Purchased': np.random.randint(1, 50, n_samples),
        'Average Rating': np.random.uniform(1, 5, n_samples).round(2),
        'Days Since Last Purchase': np.random.randint(1, 365, n_samples),
        'Satisfaction Level': np.random.choice(['Satisfied', 'Neutral', 'Unsatisfied'], n_samples),
        'Discount Applied': np.random.choice([0, 1], n_samples)
    }
    data_df = pd.DataFrame(data)
    print(f"   Generated {len(data_df)} customer records")
    
    # Clean data
    print("\n2. Cleaning data...")
    data_clean = segmenter.clean_data(data_df)
    
    # Preprocess
    print("\n3. Preprocessing data...")
    data_processed, X_scaled = segmenter.preprocess_data(data_clean)
    
    # Find optimal clusters
    print("\n4. Finding optimal number of clusters...")
    K_range, inertias, sil_scores, best_k = segmenter.find_optimal_clusters(X_scaled)
    
    # Train model
    print(f"\n5. Training model with {best_k} clusters...")
    labels = segmenter.fit_model(X_scaled, n_clusters=best_k)
    
    # Create profiles
    print("\n6. Creating cluster profiles...")
    profile_num, profile_cat, sizes = segmenter.create_cluster_profiles(data_processed, labels)
    
    print("\n" + "=" * 70)
    print("CLUSTER PROFILES")
    print("=" * 70)
    print("\nNumeric Features:")
    print(profile_num.round(2))
    print("\nCluster Sizes:")
    print(sizes)
    
    # Save model
    print("\n7. Saving model...")
    segmenter.save_model('example_model.pkl')
    
    # Make predictions for new customers
    print("\n" + "=" * 70)
    print("MAKING PREDICTIONS")
    print("=" * 70)
    
    new_customers = [
        {
            'Age': 35,
            'Gender': 'Male',
            'Membership Type': 'Gold',
            'Total Spend': 1500.00,
            'Items Purchased': 25,
            'Average Rating': 4.5,
            'City': 'New York',
            'Days Since Last Purchase': 15,
            'Satisfaction Level': 'Satisfied',
            'Discount Applied': 1
        },
        {
            'Age': 28,
            'Gender': 'Female',
            'Membership Type': 'Silver',
            'Total Spend': 300.00,
            'Items Purchased': 5,
            'Average Rating': 3.2,
            'City': 'Chicago',
            'Days Since Last Purchase': 90,
            'Satisfaction Level': 'Neutral',
            'Discount Applied': 0
        },
        {
            'Age': 45,
            'Gender': 'Male',
            'Membership Type': 'Bronze',
            'Total Spend': 150.00,
            'Items Purchased': 3,
            'Average Rating': 2.5,
            'City': 'Los Angeles',
            'Days Since Last Purchase': 200,
            'Satisfaction Level': 'Unsatisfied',
            'Discount Applied': 0
        }
    ]
    
    segment_names = {
        0: "High Spenders 💎",
        1: "Loyal Customers ⭐",
        2: "Occasional Shoppers 🛒",
        3: "Dissatisfied Customers ⚠️"
    }
    
    for i, customer in enumerate(new_customers, 1):
        print(f"\nCustomer {i}:")
        print(f"  Age: {customer['Age']}, Gender: {customer['Gender']}")
        print(f"  Total Spend: ${customer['Total Spend']:.2f}")
        print(f"  Membership: {customer['Membership Type']}")
        
        cluster = segmenter.predict_cluster(customer)
        segment_name = segment_names.get(cluster, f"Cluster {cluster}")
        
        print(f"  → Predicted Segment: {segment_name}")
    
    print("\n" + "=" * 70)


def example_2_load_and_predict():
    """Example 2: Load existing model and make predictions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Load Existing Model and Predict")
    print("=" * 70)
    
    # Load model
    print("\n1. Loading saved model...")
    segmenter = CustomerSegmentation()
    
    try:
        segmenter.load_model('example_model.pkl')
        
        # Make prediction
        print("\n2. Making prediction for new customer...")
        customer = {
            'Age': 50,
            'Gender': 'Female',
            'Membership Type': 'Gold',
            'Total Spend': 2000.00,
            'Items Purchased': 40,
            'Average Rating': 4.8,
            'City': 'New York',
            'Days Since Last Purchase': 5,
            'Satisfaction Level': 'Satisfied',
            'Discount Applied': 1
        }
        
        cluster = segmenter.predict_cluster(customer)
        
        print(f"\nCustomer Details:")
        for key, value in customer.items():
            print(f"  {key}: {value}")
        
        print(f"\n→ Predicted Cluster: {cluster}")
        
    except FileNotFoundError:
        print("   Model file not found. Please run Example 1 first.")
    
    print("\n" + "=" * 70)


def example_3_batch_predictions():
    """Example 3: Batch predictions from CSV"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Batch Predictions from CSV")
    print("=" * 70)
    
    # Load model
    print("\n1. Loading saved model...")
    segmenter = CustomerSegmentation()
    
    try:
        segmenter.load_model('example_model.pkl')
        
        # Create sample CSV data
        print("\n2. Creating sample customer data...")
        import numpy as np
        np.random.seed(123)
        
        new_customers = pd.DataFrame({
            'Customer ID': [f'NEW{i:04d}' for i in range(1, 21)],
            'Age': np.random.randint(18, 70, 20),
            'Gender': np.random.choice(['Male', 'Female'], 20),
            'City': np.random.choice(['New York', 'Los Angeles', 'Chicago'], 20),
            'Membership Type': np.random.choice(['Gold', 'Silver', 'Bronze'], 20),
            'Total Spend': np.random.gamma(2, 200, 20).round(2),
            'Items Purchased': np.random.randint(1, 50, 20),
            'Average Rating': np.random.uniform(1, 5, 20).round(2),
            'Days Since Last Purchase': np.random.randint(1, 365, 20),
            'Satisfaction Level': np.random.choice(['Satisfied', 'Neutral', 'Unsatisfied'], 20),
            'Discount Applied': np.random.choice([0, 1], 20)
        })
        
        print(f"   Created {len(new_customers)} new customer records")
        
        # Make batch predictions
        print("\n3. Making batch predictions...")
        clusters = segmenter.predict_cluster(new_customers)
        
        # Add predictions to dataframe
        new_customers['Predicted Cluster'] = clusters
        
        # Display results
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS (First 10)")
        print("=" * 70)
        print(new_customers[['Customer ID', 'Age', 'Total Spend', 'Membership Type', 'Predicted Cluster']].head(10))
        
        # Summary statistics
        print("\n" + "=" * 70)
        print("CLUSTER DISTRIBUTION")
        print("=" * 70)
        cluster_counts = new_customers['Predicted Cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            percentage = (count / len(new_customers)) * 100
            print(f"Cluster {cluster}: {count} customers ({percentage:.1f}%)")
        
        # Save results
        output_file = 'customer_predictions.csv'
        new_customers.to_csv(output_file, index=False)
        print(f"\n✅ Predictions saved to '{output_file}'")
        
    except FileNotFoundError:
        print("   Model file not found. Please run Example 1 first.")
    
    print("\n" + "=" * 70)


def example_4_custom_recommendations():
    """Example 4: Generate custom marketing recommendations"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Marketing Recommendations")
    print("=" * 70)
    
    # Define segment characteristics and recommendations
    segment_info = {
        0: {
            'name': 'High Spenders 💎',
            'characteristics': [
                'High total spend (typically >$1000)',
                'Premium membership (often Gold)',
                'Frequent purchases',
                'High satisfaction ratings'
            ],
            'marketing_strategies': [
                'Offer VIP treatment and exclusive access',
                'Provide premium products and services',
                'Personalized concierge service',
                'Early access to new products',
                'Invite to exclusive events'
            ],
            'communication': 'Premium, personalized, exclusive tone'
        },
        1: {
            'name': 'Loyal Customers ⭐',
            'characteristics': [
                'Moderate to high spend',
                'Regular purchase frequency',
                'Good satisfaction ratings',
                'Mix of membership types'
            ],
            'marketing_strategies': [
                'Loyalty rewards program',
                'Points-based incentives',
                'Regular engagement emails',
                'Birthday and anniversary offers',
                'Referral bonuses'
            ],
            'communication': 'Friendly, appreciative, engaging tone'
        },
        2: {
            'name': 'Occasional Shoppers 🛒',
            'characteristics': [
                'Lower total spend',
                'Infrequent purchases',
                'Price-sensitive',
                'Often have discounts applied'
            ],
            'marketing_strategies': [
                'Time-limited discount offers',
                'Flash sales notifications',
                'Free shipping promotions',
                'Bundle deals',
                'Cart abandonment recovery'
            ],
            'communication': 'Value-focused, promotional tone'
        },
        3: {
            'name': 'Dissatisfied Customers ⚠️',
            'characteristics': [
                'Low satisfaction ratings',
                'Declining purchase frequency',
                'May have had negative experiences',
                'At risk of churn'
            ],
            'marketing_strategies': [
                'Immediate follow-up on concerns',
                'Personalized apology and compensation',
                'Special "we miss you" offers',
                'Customer service priority',
                'Feedback surveys with incentives'
            ],
            'communication': 'Apologetic, solution-oriented, caring tone'
        }
    }
    
    # Display recommendations
    for cluster, info in segment_info.items():
        print(f"\n{'=' * 70}")
        print(f"CLUSTER {cluster}: {info['name']}")
        print('=' * 70)
        
        print("\n📊 Characteristics:")
        for char in info['characteristics']:
            print(f"  • {char}")
        
        print("\n💡 Marketing Strategies:")
        for strategy in info['marketing_strategies']:
            print(f"  ✓ {strategy}")
        
        print(f"\n📧 Communication Style: {info['communication']}")
    
    print("\n" + "=" * 70)


def main():
    """Run all examples"""
    print("\n" + "#" * 70)
    print("# CUSTOMER SEGMENTATION - USAGE EXAMPLES")
    print("#" * 70)
    
    # Run examples
    example_1_train_and_predict()
    example_2_load_and_predict()
    example_3_batch_predictions()
    example_4_custom_recommendations()
    
    print("\n" + "#" * 70)
    print("# ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("#" * 70)
    print("\nNext steps:")
    print("  1. Modify the examples for your specific use case")
    print("  2. Use your own CSV data file")
    print("  3. Customize segment names and recommendations")
    print("  4. Integrate predictions into your business workflow")
    print("\nFor interactive dashboard, run:")
    print("  streamlit run streamlit_app.py")
    print("\n")


if __name__ == "__main__":
    main()
