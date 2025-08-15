from flask import Flask, request, jsonify, session, render_template, send_from_directory, send_file, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import os
import requests
from io import BytesIO

# Dash imports
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

# Flask Setup
app = Flask(__name__, static_folder='static', )
CORS(app, supports_credentials=True, origins=["https://pandam-vilai.onrender.com"])
app.secret_key = 'your_secret_key'

# MongoDB Setup
client = MongoClient("mongodb+srv://aravindans2004:Aravindans2004@userauth.joa6bii.mongodb.net/")
db = client['market_data']
commodities = db['commodities']
db1 = client['auth_db']
users_collection = db1['users']

# Load model and column order
with open(r"models/xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open(r"models/column_order.pkl", "rb") as columns_file:
    column_order = pickle.load(columns_file)

# Historical data for dashboard
df_hist = pd.read_csv(r'filtered_apr2024_to_2025.csv')
df_hist['Arrival_Date'] = pd.to_datetime(df_hist['Arrival_Date'])
df_hist['Year'] = df_hist['Arrival_Date'].dt.year
df_hist['Month'] = df_hist['Arrival_Date'].dt.month

# COMMODITY_PRICE_RANGES omitted here for brevity — keep yours as is
COMMODITY_PRICE_RANGES = {
# Vegetables
    'Ashgourd': {"min_price_range": (1000, 1500), "max_price_range": (1800, 2500)}, # Adjusted for Ashgourd
    'Broad Beans': {"min_price_range": (2000, 3000), "max_price_range": (3000, 4000)}, # Adjusted for Broad Beans
    'Bitter Gourd': {"min_price_range": (2000, 3000), "max_price_range": (3700, 4000)}, # Adjusted for Bitter Gourd
    'Bottle Gourd': {"min_price_range": (1000, 2500), "max_price_range": (2500, 3500)}, # Adjusted for Bottle Gourd
    'Brinjal': {"min_price_range": (2000, 3300), "max_price_range": (3400, 4000)}, # Adjusted for Brinjal
    'Cabbage': {"min_price_range": (1000, 1500), "max_price_range": (1500, 2000)}, # Adjusted for Cabbage
    'Capsicum': {"min_price_range": (3000, 3800), "max_price_range": (3900, 4600)}, # Adjusted for Capsicum
    'Carrot': {"min_price_range": (3000, 3700), "max_price_range": (3700, 4200)}, # Adjusted for Carrot
    'Cluster Beans': {"min_price_range": (3000, 3700), "max_price_range": (3800, 4300)}, # Adjusted for Cluster Beans
    'Coriander (Leaves)': {"min_price_range": (500, 800), "max_price_range": (800, 1300)}, # Adjusted for Coriander
    'Cauliflower': {"min_price_range": (1800, 2800), "max_price_range": (2500, 3000)}, # Adjusted for Cauliflower
    'Drumstick': {"min_price_range": (7000, 7500), "max_price_range": (7800, 8400)}, # Adjusted for Drumstick
    'Green Chilli': {"min_price_range": (2000, 3200), "max_price_range": (3000, 4500)}, # Adjusted for Green Chilli
    'Onion': {"min_price_range": (1000, 1500), "max_price_range": (2000, 2500)}, # Adjusted for Onion
    'Potato': {"min_price_range": (2000, 2500), "max_price_range": (2500, 3500)}, # Adjusted for Potato
    'Pumpkin': {"min_price_range": (900, 1700), "max_price_range": (1800, 2400)}, # Adjusted for Pumpkin
    'Raddish': {"min_price_range": (2000, 2000), "max_price_range": (2200, 3000)}, # Adjusted for Raddish
    'Snakeguard': {"min_price_range": (1400, 2500), "max_price_range": (2700, 3400)}, # Adjusted for Snakeguard
    'Sweet Potato': {"min_price_range": (4500, 5700), "max_price_range": (5500, 6500)},
    'Tomato': {"min_price_range": (1000, 1500), "max_price_range": (1500, 2000)}, # Adjusted for Tomato
    # Pulses
    "Arhar (Tur/Red Gram)(Whole)": {"min_price_range": (2000, 4500), "max_price_range": (4800, 6000)},
    "Bengal Gram (Gram)(Whole)": {"min_price_range": (3000, 3800), "max_price_range": (4000, 4500)},
    "Bengal Gram Dal (Chana Dal)": {"min_price_range": (6000, 8000), "max_price_range": (8000, 10000)},
    "Black Gram (Urd Beans)(Whole)": {"min_price_range": (6000, 7600), "max_price_range": (7700, 8500)},
    "Black Gram Dal (Urd Dal)": {"min_price_range": (9000, 13500), "max_price_range": (13000, 15000)},
    "Green Gram (Moong)(Whole)": {"min_price_range": (6000, 7000), "max_price_range": (7000, 8000)},
    "Green Gram Dal (Moong Dal)": {"min_price_range": (8000, 9000), "max_price_range": (9200, 10000)},
    "Kabuli Chana (Chickpeas-White)": {"min_price_range": (6000, 8500), "max_price_range": (8000, 9000)},
    "Kulthi (Horse Gram)": {"min_price_range": (4000, 5300), "max_price_range": (5500, 6500)},
    "Moath Dal": {"min_price_range": (1700, 1900), "max_price_range": (1900, 2400)},
    
}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

def generate_weekly_predictions(commodity_name, num_weeks=10):
    predictions = []
    start_date = datetime.today()

    price_ranges = COMMODITY_PRICE_RANGES[commodity_name]
    min_price_range = price_ranges["min_price_range"]
    max_price_range = price_ranges["max_price_range"]

    for i in range(num_weeks):
        future_date = start_date + timedelta(weeks=i)

        future_data = {
            'Min_Price': [random.randint(*min_price_range)],
            'Max_Price': [random.randint(*max_price_range)],
            'Arrival_Year': [future_date.year],
            'Arrival_Month': [future_date.month],
            'Arrival_Day': [future_date.day]
        }

        for col in column_order:
            if col not in future_data:
                future_data[col] = [0]

        commodity_col = f'Commodity_{commodity_name}'
        if commodity_col in column_order:
            future_data[commodity_col] = [1]

        future_df = pd.DataFrame(future_data)
        future_df = future_df[column_order]

        future_price = model.predict(future_df)[0]

        predictions.append({
            'Date': future_date.strftime('%Y-%m-%d'),
            'Min_Price': future_data['Min_Price'][0],
            'Max_Price': future_data['Max_Price'][0],
            'Predicted_Modal_Price': round(float(future_price), 2),
            'Price_Per_kg': round(float(future_price) / 100, 2)
        })

    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        category = data.get("category")
        variety = data.get("variety")

        if not category or not variety:
            return jsonify({"error": "Missing 'category' or 'variety' in request"}), 400

        variety = variety.title()

        if variety not in COMMODITY_PRICE_RANGES:
            return jsonify({"error": f"Unknown commodity: {variety}"}), 400

        weekly_predictions = generate_weekly_predictions(variety, num_weeks=5)
        return jsonify({"weekly_predictions": weekly_predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<commodity>', methods=['GET'])
def predict_commodity(commodity):
    try:
        variety = commodity.title()
        if variety not in COMMODITY_PRICE_RANGES:
            return jsonify({"error": f"Unknown commodity: {variety}"}), 400

        weekly_predictions = generate_weekly_predictions(variety, num_weeks=5)
        return jsonify({
            "commodity": variety,
            "weekly_predictions": weekly_predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Dash App
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/', external_stylesheets=[dbc.themes.BOOTSTRAP])

dash_app.layout = dbc.Container([
    html.H1("Commodity Price Dashboard", style={'textAlign': 'center'}),
    html.Br(),

    dbc.Row([
        dbc.Col([
            html.Label("Select Commodity"),
            dcc.Dropdown(
                id='commodity-dropdown',
                options=[{'label': com, 'value': com} for com in df_hist['Commodity'].unique()],
                value=df_hist['Commodity'].unique()[0],
                clearable=False
            )
        ]),
        dbc.Col([
            html.Label("Select District"),
            dcc.Dropdown(
                id='district-dropdown',
                options=[{'label': dist, 'value': dist} for dist in df_hist['District'].unique()],
                value=df_hist['District'].unique()[0],
                clearable=False
            )
        ])
    ]),
    html.Br(),
    dbc.Row([dbc.Col(dcc.Graph(id='historical-line-chart'), width=12)]),
    html.Br(),
    dbc.Row([
        dbc.Col(dcc.Graph(id='bivariate-scatter'), width=6),
        dbc.Col(dcc.Graph(id='multivariate-bubble'), width=6)
    ]),
    html.Br(),
    dbc.Row([dbc.Col(dcc.Graph(id='district-bar-chart'), width=12)])
], fluid=True)

@dash_app.callback(
    Output('historical-line-chart', 'figure'),
    Output('bivariate-scatter', 'figure'),
    Output('multivariate-bubble', 'figure'),
    Output('district-bar-chart', 'figure'),
    Input('commodity-dropdown', 'value'),
    Input('district-dropdown', 'value')
)
def update_graphs(selected_commodity, selected_district):
    filtered_df = df_hist[(df_hist['Commodity'] == selected_commodity) & (df_hist['District'] == selected_district)]

    try:
        response = requests.get(f'http://127.0.0.1:5000/predict/{selected_commodity}')
        predicted_data = response.json()['weekly_predictions']
        df_pred = pd.DataFrame(predicted_data)
        df_pred['Date'] = pd.to_datetime(df_pred['Date'])
    except Exception as e:
        print(f"Prediction API error: {e}")
        df_pred = pd.DataFrame()

    fig1 = px.line(filtered_df, x='Arrival_Date', y='Modal_Price', title="Historical Modal Prices", markers=True)

    if not df_pred.empty:
        fig1.add_scatter(x=df_pred['Date'], y=df_pred['Predicted_Modal_Price'],
                         mode='lines+markers', name='Predicted')

    fig2 = px.scatter(filtered_df, x='Min_Price', y='Max_Price', trendline='ols', title="Bivariate: Min vs Max Price")
    fig3 = px.scatter(filtered_df, x='Arrival_Date', y='Modal_Price', size='Max_Price',
                      color='Year', hover_data=['District'],
                      title="Multivariate: Date vs Modal Price (Size = Max Price)")

    df_avg = df_hist[df_hist['Commodity'] == selected_commodity].groupby('District')['Modal_Price'].mean().reset_index()
    fig4 = px.bar(df_avg.sort_values(by='Modal_Price', ascending=False).head(10),
                  x='District', y='Modal_Price',
                  title="Top 10 Districts by Avg Modal Price")

    return fig1, fig2, fig3, fig4

# --- 👇 Added Download Routes ---

@app.route("/download", methods=["GET"])
def download_data():
    if "user" not in session:
        return redirect(url_for("index"))  # Redirect to login/home

    query = {}

    state = request.args.get("state")
    district = request.args.get("district")
    commodity_name = request.args.get("commodity")
    variety = request.args.get("variety")
    start_date = request.args.get("start-date")
    end_date = request.args.get("end-date")
    export_format = request.args.get("format", "csv")

    if state:
        query["state"] = state
    if district:
        query["district"] = district
    if commodity_name:
        query["commodity_name"] = commodity_name
    if start_date and end_date:
        query["date"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["date"] = {"$gte": start_date}
    elif end_date:
        query["date"] = {"$lte": end_date}

    data = list(commodities.find(query, {"_id": 0}))
    if not data:
        return "No data found for the selected filters."

    df = pd.DataFrame(data)
    today = datetime.now().strftime("%Y-%m-%d")

    if export_format == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="MarketData")
        output.seek(0)
        return send_file(output, download_name=f"market_data_{today}.xlsx", as_attachment=True)

    elif export_format == "json":
        json_data = df.to_json(orient="records", indent=4)
        output = BytesIO(json_data.encode())
        return send_file(output, download_name=f"market_data_{today}.json", as_attachment=True, mimetype="application/json")

    else:
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, download_name=f"market_data_{today}.csv", as_attachment=True, mimetype="text/csv")

@app.route("/download-page")
def download_page():
    if "user" not in session:
        return redirect(url_for("index"))
    return send_from_directory('static', 'index.html')
    

# --- End Download Routes ---

# Signup/Login routes (already present)
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')

        if users_collection.find_one({"username": username}):
            return jsonify({"error": "Username already exists"}), 400

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password
        })

        return jsonify({"message": "User registered successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        user = users_collection.find_one({"username": username})
        if not user or not check_password_hash(user['password'], password):
            return jsonify({"error": "Invalid credentials"}), 401

        session["user"] = username
        return jsonify({"message": "Login successful"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
