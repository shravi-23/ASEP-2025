# Intelligent Resource Allocation and Scheduling for Cloud Environments

## Project Overview
This project implements an intelligent resource allocation and scheduling system for cloud environments, combining hybrid time-series forecasting (LSTM + ARIMA) with Deep Q-Learning and Genetic Algorithms for optimal resource management in Kubernetes clusters.

## Key Features
- Hybrid Forecasting Engine (LSTM + ARIMA)
- Dual-stage Scheduling Optimizer (DQN + GA)
- Kubernetes Integration
- Prometheus Metrics Collection
- REST API Interface

## System Requirements
- Python 3.8+
- Kubernetes cluster (Minikube for local development)
- Docker
- Prometheus

## Project Structure
```
.
├── src/
│   ├── forecasting/         # Forecasting engine (LSTM + ARIMA)
│   ├── optimization/        # DQN and GA optimization
│   ├── kubernetes/         # Kubernetes integration
│   ├── api/               # Flask REST API
│   └── utils/             # Helper utilities
├── tests/                 # Unit and integration tests
├── kubernetes/            # K8s deployment manifests
├── notebooks/            # Jupyter notebooks for analysis
└── docs/                # Documentation
```

## Setup Instructions
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure Kubernetes cluster
4. Deploy Prometheus for metrics collection
5. Start the application:
   ```bash
   python src/main.py
   ```

## Authors
- Parth Shinge
- Radha Kulkarni
- Shravi Magdum
- Palak Mundada
- Nrusinha Mane
- Vivek Shirsath

## Project Guide
Prof. Vaishali Patil (vaishali.patil@vit.edu)

## License
MIT License

## Acknowledgments
Department of Engineering, Sciences and Humanities (DESH)
Vishwakarma Institute of Technology, Pune, Maharashtra, India #   A S E P - 2 0 2 5  
 #   A S E P - 2 0 2 5  
 #   A S E P - 2 0 2 5  
 