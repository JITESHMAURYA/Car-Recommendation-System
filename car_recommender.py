import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pickle

class CarRecommender:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.knn = None
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def load_and_clean(self):
        """Load data and basic cleaning"""
        self.df = pd.read_csv(self.csv_path)
        # Print available columns
        print(f"Available columns: {list(self.df.columns)}")
        # Remove missing values and duplicates
        self.df = self.df.dropna().drop_duplicates().reset_index(drop=True)
        print(f"✓ Loaded {len(self.df)} cars")
        return self
        
    def prepare_features(self):
        """Prepare features for KNN"""
        # Check which columns exist in the dataset
        available_cols = self.df.columns.tolist()
        
        # Define possible column names (flexible matching)
        categorical_options = {
            'brand': ['brand', 'make', 'manufacturer'],
            'fuel_type': ['fuel_type', 'fuel', 'fuel_type'],
            'transmission': ['transmission', 'transmissio', 'trans']
        }
        
        numerical_options = {
            'selling_price': ['selling_price', 'price', 'ex_showroom_price'],
            'mileage': ['mileage', 'fuel_efficiency'],
            'vehicle_age': ['vehicle_age', 'age', 'year'],
            'km_driven': ['km_driven', 'kilometers_driven', 'kms_driven'],
            'engine': ['engine', 'engine_cc'],
            'max_power': ['max_power', 'power', 'bhp'],
            'seats': ['seats', 'seating_capacity']
        }
        
        # Find matching columns
        categorical = []
        for key, options in categorical_options.items():
            for opt in options:
                if opt in available_cols:
                    categorical.append(opt)
                    break
        
        numerical = []
        for key, options in numerical_options.items():
            for opt in options:
                if opt in available_cols:
                    numerical.append(opt)
                    break
        
        print(f"Using categorical: {categorical}")
        print(f"Using numerical: {numerical}")
        
        # Encode categorical features
        for col in categorical:
            self.encoders[col] = LabelEncoder()
            self.df[f'{col}_encoded'] = self.encoders[col].fit_transform(self.df[col])
        
        # Create feature matrix
        feature_cols = numerical + [f'{c}_encoded' for c in categorical]
        features = self.df[feature_cols].values
        
        # Normalize features
        self.features = self.scaler.fit_transform(features)
        
        # Train KNN model
        self.knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn.fit(self.features)
        print(f"✓ KNN model trained with {len(feature_cols)} features")
        return self
        
    def recommend_similar(self, car_name, n=5):
        """Get similar cars by name"""
        # Find the car
        matches = self.df[self.df['car_name'].str.contains(car_name, case=False, na=False)]
        if matches.empty:
            return None
            
        idx = matches.index[0]
        selected = self.df.iloc[idx]
        
        # Find neighbors
        distances, indices = self.knn.kneighbors([self.features[idx]])
        
        # Get recommendations (skip first as it's the car itself)
        recs = self.df.iloc[indices[0][1:n+1]].copy()
        recs['similarity'] = 1 - (distances[0][1:n+1] / distances[0].max())
        
        return selected, recs
        
    def recommend_by_budget(
        self, 
        max_price,
        fuel_type=None,
        seats=None,
        min_mileage=None,
        transmission=None,
        min_power=None,
        n=5
    ):
   

        # Start with price filter
        filtered = self.df[self.df['selling_price'] <= max_price]

        # Optional filters
        if fuel_type:
            filtered = filtered[filtered['fuel_type'] == fuel_type]

        if seats:
            filtered = filtered[filtered['seats'] == seats]

        if min_mileage:
            filtered = filtered[filtered['mileage'] >= min_mileage]

        if transmission:
            filtered = filtered[filtered['transmission'] == transmission]

        if min_power:
            filtered = filtered[filtered['max_power'] >= min_power]

        # If nothing left after filtering
        if filtered.empty:
            return filtered  # return empty result

        # Compute value score (same logic as earlier)
        filtered['value'] = (
        (filtered['mileage'] / filtered['mileage'].max()) * 0.4 +
        (1 - filtered['vehicle_age'] / filtered['vehicle_age'].max()) * 0.3 +
        (1 - filtered['selling_price'] / filtered['selling_price'].max()) * 0.3
    )

        # Return top N cars
        return filtered.nlargest(n, 'value')

    def get_popular(self, n=5):
        """Get popular cars"""
        brand_counts = self.df['brand'].value_counts()
        self.df['popularity'] = self.df['brand'].map(brand_counts)
        return self.df.nlargest(n, 'popularity')
        
    def save_model(self, filepath='car_model.pkl'):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'df': self.df,
                'features': self.features,
                'knn': self.knn,
                'scaler': self.scaler,
                'encoders': self.encoders
            }, f)
        print(f"✓ Model saved to {filepath}")
        
    def load_model(self, filepath='car_model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.df = data['df']
            self.features = data['features']
            self.knn = data['knn']
            self.scaler = data['scaler']
            self.encoders = data['encoders']
        print(f"✓ Model loaded from {filepath}")


# DEMO & TESTING

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🚗 SIMPLE CAR RECOMMENDATION SYSTEM (KNN)")
    print("="*60)
    
    # Initialize and train
    recommender = CarRecommender('car_data.csv')
    recommender.load_and_clean().prepare_features()
    recommender.save_model()
    
    # Example 1: Similar cars
    print("\n" + "-"*60)
    print("EXAMPLE 1: Cars Similar to 'Swift'")
    print("-"*60)
    car, recs = recommender.recommend_similar("Swift", n=5)
    print(f"\nSelected: {car['car_name']} - ₹{car['selling_price']:,} - {car['mileage']} km/l")
    print("\nRecommendations:")
    print(recs[['car_name', 'selling_price', 'mileage', 'fuel_type', 'similarity']].to_string(index=False))
    
    # Example 2: Custom Specification-based
    print("\n" + "-"*60)
    print("EXAMPLE 2: Best Cars Under ₹5 Lakhs (Petrol)")
    print("-"*60)
    budget_cars = recommender.recommend_by_budget(500000, fuel_type='Petrol', n=5)
    print(budget_cars[['car_name', 'selling_price', 'mileage', 'vehicle_age']].to_string(index=False))
    
    # Example 3: Popular cars
    print("\n" + "-"*60)
    print("EXAMPLE 3: Most Popular Cars")
    print("-"*60)
    popular = recommender.get_popular(n=5)
    print(popular[['car_name', 'brand', 'selling_price', 'mileage']].to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ Model ready! Use it with Flask API")
    print("="*60 + "\n")