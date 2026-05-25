import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CarPreprocessor:
    def __init__(self, data_path=None):
        """
        Initializes the preprocessor. Loads the dataset and prepares the scaling/encoding configurations.
        """
        if data_path is None:
            # Resolve default path relative to this file's folder (utils/..)
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            # Parent folder of utils/ is the main project folder
            proj_dir = os.path.dirname(curr_dir)
            self.data_path = os.path.join(proj_dir, "data", "cars.csv")
        else:
            self.data_path = data_path
        self.raw_df = None
        self.df = None
        self.scaler = MinMaxScaler()
        self.numeric_cols = ['Price', 'Mileage', 'Seats', 'Engine']
        self.categorical_cols = ['Fuel Type', 'Transmission']
        
        # Unique categories loaded from data dynamically to ensure consistent one-hot dimensions
        self.fuel_types = []
        self.transmissions = []
        self.feature_names = []
        
        self.load_data()

    def load_data(self):
        """
        Loads the dataset from the CSV path, performs cleaning/parsing if required, and establishes data attributes.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Please run generate_data.py first.")
            
        self.raw_df = pd.read_csv(self.data_path)
        self.df = self.raw_df.copy()
        
        # Fill missing values if any
        self.df['Price'] = self.df['Price'].fillna(self.df['Price'].median())
        self.df['Mileage'] = self.df['Mileage'].fillna(self.df['Mileage'].median())
        self.df['Seats'] = self.df['Seats'].fillna(5).astype(int)
        self.df['Engine'] = self.df['Engine'].fillna(self.df['Engine'].median())
        self.df['Fuel Type'] = self.df['Fuel Type'].fillna('Petrol')
        self.df['Transmission'] = self.df['Transmission'].fillna('Manual')
        self.df['Company Name'] = self.df['Company Name'].fillna('Unknown')
        self.df['Car Name'] = self.df['Car Name'].fillna('Unknown Car')
        
        # Capture unique categories for one-hot encoding consistency
        self.fuel_types = sorted(list(self.df['Fuel Type'].unique()))
        self.transmissions = sorted(list(self.df['Transmission'].unique()))
        
        # Set up scaler parameters using the entire dataset
        self.scaler.fit(self.df[self.numeric_cols])

    def get_features_for_soft_recommendation(self):
        """
        Processes the entire dataset into a normalized, encoded matrix for a fully soft matching KNN search.
        Returns:
            processed_df (pd.DataFrame): Preprocessed DataFrame ready for KNN fitting.
        """
        # 1. Scale Numerical Columns
        scaled_numerics = self.scaler.transform(self.df[self.numeric_cols])
        scaled_df = pd.DataFrame(scaled_numerics, columns=self.numeric_cols)
        
        # 2. Manual One-Hot Encode Categorical Columns for robust control and clean mappings
        encoded_data = []
        for index, row in self.df.iterrows():
            row_encoded = {}
            # Encode Fuel Type
            for ft in self.fuel_types:
                row_encoded[f"Fuel_{ft}"] = 1.0 if row['Fuel Type'] == ft else 0.0
            # Encode Transmission
            for trans in self.transmissions:
                row_encoded[f"Trans_{trans}"] = 1.0 if row['Transmission'] == trans else 0.0
            encoded_data.append(row_encoded)
            
        encoded_df = pd.DataFrame(encoded_data)
        
        # Combine scaled numericals and encoded categoricals
        processed_df = pd.concat([scaled_df, encoded_df], axis=1)
        self.feature_names = list(processed_df.columns)
        
        return processed_df

    def transform_query_for_soft_recommendation(self, query):
        """
        Transforms a user input query dictionary into a preprocessed 1D vector aligned with our feature space.
        Args:
            query (dict): Contains 'Price', 'Mileage', 'Seats', 'Engine', 'Fuel Type', 'Transmission'
        Returns:
            query_vector (np.ndarray): 1xN normalized query vector.
        """
        # Scale the numerical components
        seats_val = query.get('Seats', 5)
        seats_numeric = 5 if seats_val == 'All' or seats_val is None else int(seats_val)
        
        numeric_df = pd.DataFrame([[
            query.get('Price', self.df['Price'].median()),
            query.get('Mileage', self.df['Mileage'].median()),
            seats_numeric,
            query.get('Engine', self.df['Engine'].median())
        ]], columns=self.numeric_cols)
        
        scaled_numeric = self.scaler.transform(numeric_df)[0]
        
        # Construct the query dictionary
        query_vector_dict = {
            'Price': scaled_numeric[0],
            'Mileage': scaled_numeric[1],
            'Seats': scaled_numeric[2],
            'Engine': scaled_numeric[3]
        }
        
        # Add encoded categoricals
        for ft in self.fuel_types:
            query_vector_dict[f"Fuel_{ft}"] = 1.0 if query.get('Fuel Type') == ft else 0.0
            
        for trans in self.transmissions:
            query_vector_dict[f"Trans_{trans}"] = 1.0 if query.get('Transmission') == trans else 0.0
            
        # Build vector matching feature order exactly
        vector = [query_vector_dict[feat] for feat in self.feature_names]
        return np.array(vector).reshape(1, -1)

    def get_features_for_strict_recommendation(self, filtered_df):
        """
        For strict mode, we filter the dataset first by categorical criteria, and then perform KNN purely on
        numerical columns. This scales the numerical columns based on the original scaler.
        Args:
            filtered_df (pd.DataFrame): Already filtered subset of the main DataFrame
        Returns:
            scaled_numerics (pd.DataFrame): Scaled numerical features of the filtered subset.
        """
        if filtered_df.empty:
            return pd.DataFrame()
            
        scaled_numerics = self.scaler.transform(filtered_df[self.numeric_cols])
        return pd.DataFrame(scaled_numerics, columns=self.numeric_cols, index=filtered_df.index)

    def transform_query_for_strict_recommendation(self, query):
        """
        Transforms a query to a numerical vector for strict KNN matching.
        """
        seats_val = query.get('Seats', 5)
        seats_numeric = 5 if seats_val == 'All' or seats_val is None else int(seats_val)
        
        numeric_df = pd.DataFrame([[
            query.get('Price', self.df['Price'].median()),
            query.get('Mileage', self.df['Mileage'].median()),
            seats_numeric,
            query.get('Engine', self.df['Engine'].median())
        ]], columns=self.numeric_cols)
        
        scaled_numeric = self.scaler.transform(numeric_df)
        return scaled_numeric # 1x4 vector
