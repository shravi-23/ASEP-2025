import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create the sample_data directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'sample_data'), exist_ok=True)

def generate_timestamp_range(hours=168):  # 1 week of hourly data
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=hours)
    return pd.date_range(start=start_time, end=end_time, freq='h')

def add_seasonal_pattern(base, amplitude=10, period=24):
    return amplitude * np.sin(2 * np.pi * np.arange(len(base)) / period)

def add_trend(base, slope=0.1):
    return np.arange(len(base)) * slope

def add_noise(base, scale=5):
    return np.random.normal(0, scale, len(base))

def generate_ecommerce_workload():
    timestamps = generate_timestamp_range()
    base_load = 50 + add_seasonal_pattern(timestamps, amplitude=20)
    trend = add_trend(timestamps, slope=0.05)
    noise = add_noise(timestamps, scale=10)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': np.clip(base_load + trend + noise, 0, 100),
        'memory_usage': np.clip(base_load * 0.8 + trend * 0.5 + noise, 0, 100),
        'network_traffic': np.clip(base_load * 1.2 + trend * 0.3 + noise * 1.5, 0, 100),
        'response_time': np.clip(base_load * 0.3 + noise * 0.5, 0, 1000),
        'active_users': np.clip((base_load + trend + noise) * 10, 0, 1000).astype(int)
    })
    return data

def generate_batch_processing():
    timestamps = generate_timestamp_range()
    base_pattern = np.zeros(len(timestamps))
    
    # Add periodic batch jobs
    for i in range(0, len(timestamps), 24):
        if i + 4 <= len(timestamps):
            base_pattern[i:i+4] = 90
    
    noise = add_noise(timestamps, scale=5)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': np.clip(base_pattern + noise, 0, 100),
        'memory_usage': np.clip(base_pattern * 0.9 + noise, 0, 100),
        'disk_io': np.clip(base_pattern * 1.1 + noise * 2, 0, 100),
        'queue_length': np.clip(base_pattern * 0.5 + noise, 0, 1000).astype(int),
        'job_count': np.clip(base_pattern * 0.3 + noise, 0, 100).astype(int)
    })
    return data

def generate_ml_training():
    timestamps = generate_timestamp_range(hours=72)  # 3 days
    base_load = np.ones(len(timestamps)) * 85
    noise = add_noise(timestamps, scale=10)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'gpu_usage': np.clip(base_load + noise, 0, 100),
        'cpu_usage': np.clip(base_load * 0.6 + noise, 0, 100),
        'memory_usage': np.clip(base_load * 0.8 + noise, 0, 100),
        'training_loss': np.exp(-np.arange(len(timestamps)) * 0.01) + np.abs(noise) * 0.1,
        'batch_size': np.ones(len(timestamps)) * 64
    })
    return data

def generate_microservices():
    timestamps = generate_timestamp_range()
    base_load = 40 + add_seasonal_pattern(timestamps, amplitude=30)
    noise = add_noise(timestamps, scale=15)
    
    services = ['auth', 'api', 'database', 'cache', 'worker']
    data = pd.DataFrame({'timestamp': timestamps})
    
    for service in services:
        service_noise = add_noise(timestamps, scale=10)
        data[f'{service}_cpu'] = np.clip(base_load + service_noise, 0, 100)
        data[f'{service}_memory'] = np.clip(base_load * 0.7 + service_noise, 0, 100)
        data[f'{service}_latency'] = np.clip(base_load * 0.2 + service_noise, 0, 1000)
    
    return data

def generate_cdn_traffic():
    timestamps = generate_timestamp_range()
    base_load = 60 + add_seasonal_pattern(timestamps, amplitude=25)
    noise = add_noise(timestamps, scale=10)
    
    regions = ['us_east', 'us_west', 'eu_central', 'asia_east']
    data = pd.DataFrame({'timestamp': timestamps})
    
    for region in regions:
        region_noise = add_noise(timestamps, scale=8)
        data[f'{region}_bandwidth'] = np.clip(base_load + region_noise, 0, 1000)
        data[f'{region}_requests'] = np.clip((base_load + region_noise) * 100, 0, 10000).astype(int)
        data[f'{region}_cache_hit'] = np.clip(base_load + region_noise, 0, 100)
    
    return data

def generate_database_metrics():
    timestamps = generate_timestamp_range()
    base_load = 45 + add_seasonal_pattern(timestamps, amplitude=15)
    noise = add_noise(timestamps, scale=7)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': np.clip(base_load + noise, 0, 100),
        'memory_usage': np.clip(base_load * 1.2 + noise, 0, 100),
        'disk_usage': np.clip(base_load * 0.9 + add_trend(timestamps, 0.02) + noise, 0, 100),
        'active_connections': np.clip((base_load + noise) * 5, 0, 1000).astype(int),
        'queries_per_second': np.clip((base_load + noise) * 10, 0, 5000).astype(int),
        'cache_hit_ratio': np.clip(80 + noise * 0.5, 0, 100)
    })
    return data

def generate_container_metrics():
    timestamps = generate_timestamp_range()
    base_load = 55 + add_seasonal_pattern(timestamps, amplitude=20)
    noise = add_noise(timestamps, scale=12)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'running_containers': np.clip((base_load + noise) * 0.5, 0, 100).astype(int),
        'cpu_usage': np.clip(base_load + noise, 0, 100),
        'memory_usage': np.clip(base_load * 0.85 + noise, 0, 100),
        'network_in': np.clip((base_load + noise) * 100, 0, 10000),
        'network_out': np.clip((base_load + noise) * 80, 0, 8000),
        'disk_read': np.clip((base_load + noise) * 50, 0, 5000),
        'disk_write': np.clip((base_load + noise) * 30, 0, 3000)
    })
    return data

def generate_serverless_functions():
    timestamps = generate_timestamp_range()
    base_load = 30 + add_seasonal_pattern(timestamps, amplitude=40)
    noise = add_noise(timestamps, scale=15)
    
    functions = ['image_processing', 'data_transformation', 'notification', 'authentication']
    data = pd.DataFrame({'timestamp': timestamps})
    
    for function in functions:
        fn_noise = add_noise(timestamps, scale=10)
        data[f'{function}_invocations'] = np.clip((base_load + fn_noise) * 5, 0, 1000).astype(int)
        data[f'{function}_duration'] = np.clip(base_load + fn_noise, 0, 1000)
        data[f'{function}_memory'] = np.clip(base_load * 0.5 + fn_noise, 0, 512)
        data[f'{function}_errors'] = np.clip(fn_noise * 0.1, 0, 100).astype(int)
    
    return data

def generate_gaming_server():
    timestamps = generate_timestamp_range()
    base_load = 70 + add_seasonal_pattern(timestamps, amplitude=25)
    noise = add_noise(timestamps, scale=10)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'active_players': np.clip((base_load + noise) * 10, 0, 1000).astype(int),
        'cpu_usage': np.clip(base_load + noise, 0, 100),
        'memory_usage': np.clip(base_load * 0.9 + noise, 0, 100),
        'network_latency': np.clip(20 + noise * 0.5, 0, 200),
        'matches_running': np.clip((base_load + noise) * 0.2, 0, 100).astype(int),
        'matchmaking_time': np.clip(base_load * 0.1 + noise * 0.2, 0, 60)
    })
    return data

def generate_video_streaming():
    timestamps = generate_timestamp_range()
    base_load = 65 + add_seasonal_pattern(timestamps, amplitude=30)
    noise = add_noise(timestamps, scale=12)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'concurrent_streams': np.clip((base_load + noise) * 15, 0, 2000).astype(int),
        'bandwidth_usage': np.clip((base_load + noise) * 100, 0, 10000),
        'transcoding_queue': np.clip((base_load + noise) * 0.3, 0, 100).astype(int),
        'storage_used': np.clip(base_load * 0.8 + add_trend(timestamps, 0.05) + noise, 0, 100),
        'cpu_usage': np.clip(base_load + noise, 0, 100),
        'memory_usage': np.clip(base_load * 0.75 + noise, 0, 100)
    })
    return data

def main():
    # Generate all datasets
    datasets = {
        'ecommerce_workload': generate_ecommerce_workload(),
        'batch_processing': generate_batch_processing(),
        'ml_training': generate_ml_training(),
        'microservices': generate_microservices(),
        'cdn_traffic': generate_cdn_traffic(),
        'database_metrics': generate_database_metrics(),
        'container_metrics': generate_container_metrics(),
        'serverless_functions': generate_serverless_functions(),
        'gaming_server': generate_gaming_server(),
        'video_streaming': generate_video_streaming()
    }
    
    # Save datasets to CSV files
    data_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    for name, df in datasets.items():
        output_path = os.path.join(data_dir, f'{name}.csv')
        df.to_csv(output_path, index=False)
        print(f"Generated {name}.csv with {len(df)} records")

if __name__ == "__main__":
    main() 