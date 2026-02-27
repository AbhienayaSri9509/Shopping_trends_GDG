import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.title("🛍 Shopper Behavior Intelligence Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload shopping_trends.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)

    # Feature Engineering
    rfm = df.groupby('Customer ID').agg({
        'Item Purchased': 'count',
        'Purchase Amount (USD)': 'sum',
        'Review Rating': 'mean',
        'Category': 'nunique'
    })

    rfm.columns = ['Frequency','Monetary','Avg_Rating','Category_Diversity']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    df = df.merge(rfm[['Cluster']],
                  left_on='Customer ID',
                  right_index=True)

    st.subheader("Customer Segment Distribution")
    st.bar_chart(rfm['Cluster'].value_counts())

    st.subheader("Segment Profiles")
    st.dataframe(rfm.groupby('Cluster').mean())

    st.subheader("Segment vs Category")
    segment_category = df.groupby(['Cluster','Category']).size().unstack().fillna(0)
    st.dataframe(segment_category)

    st.subheader("Top Affinity Rules")

    basket = df.groupby(['Customer ID','Category'])['Category'] \
               .count().unstack().fillna(0)

    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    frequent_itemsets = apriori(basket,
                                min_support=0.05,
                                use_colnames=True)

    rules = association_rules(frequent_itemsets,
                              metric="lift",
                              min_threshold=1)

    st.dataframe(rules[['antecedents','consequents','lift']].head())