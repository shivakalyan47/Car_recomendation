import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def recommend_cars(preprocessor, query, top_n=5, mode='Strict', metric='euclidean'):
    """
    Core KNN recommendation engine.
    Args:
        preprocessor (CarPreprocessor): Initialized preprocessor instance.
        query (dict): User preferences dictionary containing:
                     'Price' (Budget), 'Mileage', 'Seats', 'Engine', 'Fuel Type', 'Transmission'
        top_n (int): Number of recommendations to return.
        mode (str): 'Strict' or 'Soft' recommendation strategy.
        metric (str): 'euclidean' or 'cosine' distance metric.
    Returns:
        recommendations (pd.DataFrame): DataFrame of top_n recommended cars with distance and confidence scores.
    """
    raw_data = preprocessor.df.copy()
    
    if mode == 'Strict':
        # 1. Apply Strict Filtering on Categorical Fields first
        filtered_df = raw_data.copy()
        
        # Apply Fuel Type filter if specified and not 'All'
        if query.get('Fuel Type') and query.get('Fuel Type') != 'All':
            filtered_df = filtered_df[filtered_df['Fuel Type'] == query.get('Fuel Type')]
            
        # Apply Transmission filter if specified and not 'All'
        if query.get('Transmission') and query.get('Transmission') != 'All':
            filtered_df = filtered_df[filtered_df['Transmission'] == query.get('Transmission')]
            
        # Apply Seating Capacity filter if specified and not 'All'
        if query.get('Seats') and query.get('Seats') != 'All':
            filtered_df = filtered_df[filtered_df['Seats'] == int(query.get('Seats'))]
            
        # Check if the filtered set is empty
        if filtered_df.empty:
            return pd.DataFrame()  # Return empty DataFrame if no exact match exists
            
        # 2. Extract and Scale numerical columns of the filtered set
        scaled_features = preprocessor.get_features_for_strict_recommendation(filtered_df)
        
        # Get query vector (numerical only)
        query_vector = preprocessor.transform_query_for_strict_recommendation(query)
        
        # 3. Fit KNN on numeric features only
        # Adjust n_neighbors if the filtered set has fewer rows than top_n
        n_neighbors = min(len(scaled_features), top_n)
        
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='brute')
        knn.fit(scaled_features.values)
        
        # 4. Search nearest neighbors
        distances, indices = knn.kneighbors(query_vector)
        
        # 5. Extract and format results
        recommended_indices = scaled_features.index[indices[0]]
        recommended_cars = raw_data.loc[recommended_indices].copy()
        recommended_cars['Distance'] = distances[0]
        
        # 6. Calculate Confidence Score
        # For Euclidean, max distance of 4 scaled numeric columns is sqrt(4) = 2.0
        if metric == 'euclidean':
            max_dist = np.sqrt(4)
            recommended_cars['Confidence Score'] = recommended_cars['Distance'].apply(
                lambda d: max(0.0, min(100.0, round(100 * (1 - (d / max_dist)), 2)))
            )
        else: # Cosine similarity
            # Cosine distance ranges from 0 to 1 for non-negative vectors
            recommended_cars['Confidence Score'] = recommended_cars['Distance'].apply(
                lambda d: max(0.0, min(100.0, round(100 * (1 - d), 2)))
            )
            
        return recommended_cars.sort_values(by='Confidence Score', ascending=False)
        
    else: # Soft Mode
        # 1. Process entire dataset (numerical + categorical)
        scaled_features = preprocessor.get_features_for_soft_recommendation()
        
        # Transform full query vector (numerical + categorical)
        query_vector = preprocessor.transform_query_for_soft_recommendation(query)
        
        # 2. Fit KNN on full preprocessed matrix
        knn = NearestNeighbors(n_neighbors=top_n, metric=metric, algorithm='brute')
        knn.fit(scaled_features.values)
        
        # 3. Search nearest neighbors
        distances, indices = knn.kneighbors(query_vector)
        
        # 4. Extract and format results
        recommended_cars = raw_data.loc[indices[0]].copy()
        recommended_cars['Distance'] = distances[0]
        
        # 5. Calculate Confidence Score
        # In soft mode, number of dimensions = 4 (numeric) + len(fuel_types) + len(transmissions)
        num_features = scaled_features.shape[1]
        
        if metric == 'euclidean':
            max_dist = np.sqrt(num_features)
            recommended_cars['Confidence Score'] = recommended_cars['Distance'].apply(
                lambda d: max(0.0, min(100.0, round(100 * (1 - (d / max_dist)), 2)))
            )
        else: # Cosine similarity
            recommended_cars['Confidence Score'] = recommended_cars['Distance'].apply(
                lambda d: max(0.0, min(100.0, round(100 * (1 - d), 2)))
            )
            
        return recommended_cars.sort_values(by='Confidence Score', ascending=False)
