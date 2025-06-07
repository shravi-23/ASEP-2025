import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from standalone import StandaloneOptimizer
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize optimizer
optimizer = StandaloneOptimizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
        
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Validate required columns
        required_columns = ['timestamp', 'cpu_utilization', 'memory_utilization', 'running_pods', 'total_pods']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}'
            }), 400
            
        # Train the models
        optimizer.train_models(df)
        
        # Run optimization
        results = optimizer.optimize_resources()
        
        # Prepare visualization data
        time_points = df['timestamp'].iloc[::4].tolist()  # Sample every 4th point
        cpu_util = df['cpu_utilization'].iloc[::4].multiply(100).round(1).tolist()
        mem_util = df['memory_utilization'].iloc[::4].multiply(100).round(1).tolist()
        running_pods = int(df['running_pods'].iloc[-1])
        available_pods = int(df['total_pods'].iloc[-1]) - running_pods
        
        # Format results for display
        formatted_results = {
            'cpu_request': results['cpu_request'],
            'memory_request': results['memory_request'],
            'replicas': results['replicas'],
            'predicted_utilization': f"{results['predicted_utilization']:.2%}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'visualization_data': {
                'utilization': {
                    'labels': time_points,
                    'cpu': cpu_util,
                    'memory': mem_util
                },
                'pods': {
                    'running': running_pods,
                    'available': available_pods
                }
            }
        }
        
        return jsonify({
            'success': True,
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        results = optimizer.optimize_resources()
        
        formatted_results = {
            'cpu_request': results['cpu_request'],
            'memory_request': results['memory_request'],
            'replicas': results['replicas'],
            'predicted_utilization': f"{results['predicted_utilization']:.2%}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({
            'success': True,
            'results': formatted_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 