import pandas as pd
import numpy as np

def recommend_cars_nb(preprocessor, query, top_n=5, decay_factor=2.5):
    """
    Core Naive Bayes recommendation engine.
    Computes the posterior probability of each car matching the user's preferences.
    
    Args:
        preprocessor (CarPreprocessor): Initialized preprocessor instance.
        query (dict): User preferences dictionary containing:
                     'Price' (Budget), 'Mileage', 'Seats', 'Engine', 'Fuel Type', 'Transmission'
        top_n (int): Number of recommendations to return.
        decay_factor (float): Divisor for exponent to scale match confidence decay.
        
    Returns:
        recommendations (pd.DataFrame): DataFrame of top_n recommended cars with confidence scores.
    """
    df = preprocessor.df.copy()
    
    # 1. Compute standard deviations of continuous columns from the entire dataset
    # This serves as the bandwidth of our Gaussian Likelihoods.
    price_std = max(df['Price'].std(), 0.1)
    mileage_std = max(df['Mileage'].std(), 0.1)
    engine_std = max(df['Engine'].std(), 10.0)
    
    # User query preferences
    q_price = float(query.get('Price', df['Price'].median()))
    q_mileage = float(query.get('Mileage', df['Mileage'].median()))
    q_engine = float(query.get('Engine', 1200.0))
    q_fuel = query.get('Fuel Type', 'All')
    q_trans = query.get('Transmission', 'All')
    
    # Handle seats: can be 'All' or a string representing integer
    q_seats = query.get('Seats', 'All')
    if q_seats != 'All' and q_seats is not None:
        q_seats = int(q_seats)

    log_likelihoods = []
    
    for idx, row in df.iterrows():
        log_L = 0.0
        log_L_max = 0.0
        
        # --- PRICE (Gaussian continuous likelihood) ---
        # Premium business logic: being UNDER budget is a good thing! We only penalize if it exceeds the budget.
        if row['Price'] <= q_price:
            p_price_log = 0.0
            p_price_max_log = 0.0
        else:
            variance = price_std ** 2
            p_price_log = -0.5 * np.log(2 * np.pi * variance) - ((q_price - row['Price']) ** 2) / (2 * variance)
            p_price_max_log = -0.5 * np.log(2 * np.pi * variance)
            
        log_L += p_price_log
        log_L_max += p_price_max_log
        
        # --- MILEAGE (Gaussian continuous likelihood) ---
        # Premium business logic: exceeding the user's minimum target efficiency is a good thing! We only penalize if below.
        if row['Mileage'] >= q_mileage:
            p_mileage_log = 0.0
            p_mileage_max_log = 0.0
        else:
            variance = mileage_std ** 2
            p_mileage_log = -0.5 * np.log(2 * np.pi * variance) - ((q_mileage - row['Mileage']) ** 2) / (2 * variance)
            p_mileage_max_log = -0.5 * np.log(2 * np.pi * variance)
            
        log_L += p_mileage_log
        log_L_max += p_mileage_max_log
        
        # --- ENGINE CAPACITY (Gaussian continuous likelihood) ---
        # Exclude electric vehicles from engine displacement penalty
        if row['Fuel Type'] != 'Electric' and q_fuel != 'Electric':
            variance = engine_std ** 2
            p_engine_log = -0.5 * np.log(2 * np.pi * variance) - ((q_engine - row['Engine']) ** 2) / (2 * variance)
            p_engine_max_log = -0.5 * np.log(2 * np.pi * variance)
            
            log_L += p_engine_log
            log_L_max += p_engine_max_log
            
        # --- CATEGORICAL: FUEL TYPE ---
        if q_fuel != 'All':
            if row['Fuel Type'] == q_fuel:
                log_L += np.log(0.95)
            else:
                log_L += np.log(0.05)
            log_L_max += np.log(0.95)
            
        # --- CATEGORICAL: TRANSMISSION ---
        if q_trans != 'All':
            if row['Transmission'] == q_trans:
                log_L += np.log(0.95)
            else:
                log_L += np.log(0.05)
            log_L_max += np.log(0.95)
            
        # --- CATEGORICAL: SEATS ---
        if q_seats != 'All':
            if int(row['Seats']) == int(q_seats):
                log_L += np.log(0.95)
            else:
                log_L += np.log(0.05)
            log_L_max += np.log(0.95)
            
        # Calculate log-difference between perfect match and actual match
        log_diff = log_L_max - log_L
        
        # Convert to 0-100 Confidence Score using decay factor
        conf_score = 100.0 * np.exp(-log_diff / decay_factor)
        
        log_likelihoods.append({
            'index': idx,
            'Confidence Score': round(conf_score, 2),
            'Distance': round(log_diff, 4)
        })
        
    results_df = pd.DataFrame(log_likelihoods)
    results_df.set_index('index', inplace=True)
    
    # Merge and sort
    recommended_cars = df.join(results_df).copy()
    return recommended_cars.sort_values(by='Confidence Score', ascending=False).head(top_n)
