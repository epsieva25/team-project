import random
from flask import Flask, render_template, request
import joblib
from faker import Faker
import plotly.graph_objects as go
import numpy as np
import xgboost as xgb  # Import XGBoost

app = Flask(__name__)

fake = Faker()
test_history = []

# Load the model and scaler
model = joblib.load('model/optimized_network_model.pkl')
scaler = joblib.load('model/scaler.pkl')

def get_network_data():
    """
    Generate random network data for testing.
    """
    download_speed = round(random.uniform(10, 500), 2)
    upload_speed = round(random.uniform(5, 100), 2)
    ping = random.randint(10, 150)
    network_type = random.choice(["Wi-Fi", "Ethernet", "4G", "5G"])
    real_time_latency = random.randint(10, 200)
    network_uptime = f"{random.randint(1, 48)} hours"
    connection_status = random.choice(["Connected", "Disconnected"])
    total_disconnects = random.randint(0, 5)
    data_usage = {'sent_mb': random.randint(50, 200), 'received_mb': random.randint(50, 200)}
    public_ip = fake.ipv4()
    isp = fake.company()
    vpn = random.choice(["Yes", "No"])
    
    return {
        'download_speed_mbps': download_speed,
        'upload_speed_mbps': upload_speed,
        'ping_ms': ping,
        'network_type': network_type,
        'real_time_latency_ms': real_time_latency,
        'network_uptime': network_uptime,
        'connection_status': connection_status,
        'total_disconnects': total_disconnects,
        'data_usage': data_usage,
        'public_ip_info': {'public_ip': public_ip, 'isp': isp, 'vpn': vpn}
    }

def get_multiple_network_data(num_tests=5):
    """
    Generate multiple network data points and calculate averages.
    """
    data = []
    for _ in range(num_tests):
        data.append(get_network_data())
    
    avg_download_speed = round(sum(d['download_speed_mbps'] for d in data) / num_tests, 2)
    avg_upload_speed = round(sum(d['upload_speed_mbps'] for d in data) / num_tests, 2)
    
    return data, avg_download_speed, avg_upload_speed

def create_interactive_graph():
    """
    Create an interactive graph for download and upload speeds.
    """
    tests = [test['data'] for test in test_history]
    download_speeds = [test['download_speed_mbps'] for test in sum(tests, [])]
    upload_speeds = [test['upload_speed_mbps'] for test in sum(tests, [])]
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(download_speeds) + 1)),  # Fixed: Added closing parenthesis
        y=download_speeds,
        mode='lines+markers',
        name='Download Speed (Mbps)',
        line=dict(color='blue', width=3),
        marker=dict(color='blue', size=8, line=dict(color='black', width=1)),
    ))

    fig.add_trace(go.Scatter(
        x=list(range(1, len(upload_speeds) + 1)),  # Fixed: Added closing parenthesis
        y=upload_speeds,
        mode='lines+markers',
        name='Upload Speed (Mbps)',
        line=dict(color='red', width=3),
        marker=dict(color='red', size=8, line=dict(color='black', width=1)),
    ))

    fig.update_layout(
        title="Network Speed Trends Over Time",
        xaxis_title="Test Number",
        yaxis_title="Speed (Mbps)",
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=True
    )

    graph_html = fig.to_html(full_html=False)
    return graph_html

def predict_connection_status(data):
    """
    Predict the connection status using the trained XGBoost model.
    """
    # Define the required features (13 features as expected by the scaler)
    required_features = [
        'download_speed_mbps', 'upload_speed_mbps', 'ping_ms', 'network_uptime', 'total_disconnects', 'network_type',
        'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13'
    ]

    # Prepare input data for the model
    input_data = []
    for feature in required_features:
        if feature == 'network_uptime':
            # Convert uptime string to numeric value (e.g., "24 hours" -> 24)
            uptime = int(data[feature].split()[0])
            input_data.append(uptime)
        elif feature == 'network_type':
            # Convert network type to numeric (e.g., "Wi-Fi" -> 0, "Ethernet" -> 1, etc.)
            network_types = ["Wi-Fi", "Ethernet", "4G", "5G"]
            input_data.append(network_types.index(data[feature]))
        elif feature in data:
            input_data.append(data[feature])
        else:
            # Add dummy value for missing features
            input_data.append(0)

    # Scale the input data
    input_data_scaled = scaler.transform([input_data])

    # Convert the scaled data to a DMatrix object (required by XGBoost)
    dmatrix = xgb.DMatrix(input_data_scaled)

    # Predict the connection status
    predicted_connection_status = model.predict(dmatrix)
    
    return "Connected" if predicted_connection_status[0] == 1 else "Disconnected"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/result', methods=['POST'])
def result():
    """
    Handle the result page and display network data.
    """
    result_data, avg_download_speed, avg_upload_speed = get_multiple_network_data()
    test_history.append({'data': result_data, 'avg_download_speed': avg_download_speed, 'avg_upload_speed': avg_upload_speed})
    

    if len(test_history) > 5:
        test_history.pop(0)
    
    for data in result_data:
        data['predicted_connection_status'] = predict_connection_status(data)
    
    graph_html = create_interactive_graph()
    
    return render_template('result.html', result=result_data, graph_html=graph_html, 
                           avg_download_speed=avg_download_speed, avg_upload_speed=avg_upload_speed)

if __name__ == '__main__':
    app.run(debug=True)