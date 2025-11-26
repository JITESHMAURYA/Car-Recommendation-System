"""
Simple Flask API for Car Recommendation System
Beginner-friendly version - Under 100 lines
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from car_recommender import CarRecommender
import os

app = Flask(__name__)
CORS(app)

# Load model
recommender = CarRecommender('car_data.csv')
if os.path.exists('car_model.pkl'):
    recommender.load_model()
else:
    recommender.load_and_clean().prepare_features().save_model()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """API documentation"""
    return {
        'message': '🚗 Car Recommendation API',
        'endpoints': {
            'GET /stats': 'Get dataset statistics',
            'POST /recommend/similar': 'Get similar cars {"car_name": "Swift", "n": 5}',
            'POST /recommend/custom': 'Get cars by custom specs {"max_price": 500000, "fuel_type": "Petrol", "min_seats": 5, "min_mileage": 15, "min_engine": 800, "transmission": "Manual", "n": 10}',
            'GET /recommend/popular': 'Get popular cars (?n=5)'
        }
    }

@app.route('/stats')
def stats():
    """Dataset statistics"""
    df = recommender.df
    return jsonify({
        'total_cars': len(df),
        'brands': df['brand'].nunique(),
        'avg_price': float(df['selling_price'].mean()),
        'price_range': {'min': int(df['selling_price'].min()), 'max': int(df['selling_price'].max())},
        'fuel_types': df['fuel_type'].value_counts().to_dict()
    })

@app.route('/recommend/similar', methods=['POST'])
def recommend_similar():
    """Get similar cars"""
    data = request.json
    car_name = data.get('car_name')
    n = data.get('n', 5)
    
    if not car_name:
        return jsonify({'error': 'car_name required'}), 400
    
    result = recommender.recommend_similar(car_name, n)
    if result is None:
        return jsonify({'error': 'Car not found'}), 404
    
    car, recs = result
    return jsonify({
        'selected': car.to_dict(),
        'recommendations': recs.to_dict('records')
    })

@app.route('/recommend/custom', methods=['POST'])
def recommend_custom():
    """Get cars by custom specifications"""
    data = request.json
    max_price = data.get('max_price')
    fuel_type = data.get('fuel_type')
    min_seats = data.get('min_seats')
    min_mileage = data.get('min_mileage')
    min_engine = data.get('min_engine')
    transmission = data.get('transmission')
    n = data.get('n', 10)
    
    if not max_price:
        return jsonify({'error': 'max_price required'}), 400
    
    # Start with all cars
    df = recommender.df.copy()
    
    # Apply filters
    df = df[df['selling_price'] <= max_price]
    
    if fuel_type:
        df = df[df['fuel_type'] == fuel_type]
    
    if min_seats:
        df = df[df['seats'] >= min_seats]
    
    if min_mileage:
        df = df[df['mileage'] >= min_mileage]
    
    if min_engine:
        df = df[df['engine'] >= min_engine]
    
    if transmission:
        # Check if transmission column exists
        if 'transmission' in df.columns or 'transmissio' in df.columns:
            trans_col = 'transmission' if 'transmission' in df.columns else 'transmissio'
            df = df[df[trans_col] == transmission]
    
    # Sort by value score if available, otherwise by mileage
    if 'value' in df.columns:
        df = df.sort_values('value', ascending=False)
    else:
        df = df.sort_values('mileage', ascending=False)
    
    recs = df.head(n)
    
    return jsonify({
        'count': len(recs),
        'recommendations': recs.to_dict('records')
    })

@app.route('/recommend/popular')
def recommend_popular():
    """Get popular cars"""
    n = request.args.get('n', default=5, type=int)
    popular = recommender.get_popular(n)
    return jsonify({
        'count': len(popular),
        'recommendations': popular.to_dict('records')
    })

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("\n🚗 Car Recommendation API")
    print("Server: http://localhost:5000")
    print("Docs: http://localhost:5000\n")
    app.run(debug=True, port=5000)