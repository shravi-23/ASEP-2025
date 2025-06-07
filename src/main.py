import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Tuple

from forecasting.hybrid_model import HybridForecastingModel
from optimization.dqn_agent import DQNAgent
from optimization.genetic_algorithm import GeneticAlgorithm
from kubernetes.k8s_manager import KubernetesManager

class ResourceOptimizer:
    def __init__(
        self,
        forecast_sequence_length: int = 10,
        dqn_state_size: int = 6,
        dqn_action_size: int = 5,
        ga_population_size: int = 50,
        ga_chromosome_length: int = 3
    ):
        # Initialize components
        self.forecasting_model = HybridForecastingModel(
            sequence_length=forecast_sequence_length
        )
        self.dqn_agent = DQNAgent(
            state_size=dqn_state_size,
            action_size=dqn_action_size
        )
        self.k8s_manager = KubernetesManager()
        
        # Initialize GA with a fitness function that combines cost and performance
        def fitness_function(chromosome: np.ndarray) -> float:
            cpu_allocation = chromosome[0]
            memory_allocation = chromosome[1]
            replicas = chromosome[2]
            
            # Calculate cost component
            cost = (cpu_allocation * memory_allocation * replicas) / 100.0
            
            # Calculate performance component (higher utilization is better)
            cluster_state = self.k8s_manager.get_cluster_state()
            cpu_utilization = cluster_state[0]
            memory_utilization = cluster_state[1]
            performance = (cpu_utilization + memory_utilization) / 2.0
            
            # Combine metrics (70% performance, 30% cost efficiency)
            return 0.7 * performance - 0.3 * cost
            
        self.genetic_algorithm = GeneticAlgorithm(
            population_size=ga_population_size,
            chromosome_length=ga_chromosome_length,
            fitness_func=fitness_function
        )
        
        # Training parameters
        self.batch_size = 32
        self.update_interval = 5  # minutes
        
    def collect_metrics_history(self, hours: int = 24) -> pd.DataFrame:
        """Collect historical metrics for training"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        metrics = []
        
        current_time = start_time
        while current_time <= end_time:
            state = self.k8s_manager.get_cluster_state()
            metrics.append({
                'timestamp': current_time,
                'cpu_utilization': state[0],
                'memory_utilization': state[1],
                'running_pods': state[2],
                'total_pods': state[3]
            })
            current_time += timedelta(minutes=5)
            
        return pd.DataFrame(metrics)
        
    def train_models(self, training_data: pd.DataFrame) -> None:
        """Train both forecasting and optimization models"""
        # Train forecasting model
        self.forecasting_model.fit(
            training_data['cpu_utilization'],
            arima_order=(2, 1, 2)
        )
        
        # Train DQN agent on historical data
        for i in range(len(training_data) - 1):
            state = training_data.iloc[i][['cpu_utilization', 'memory_utilization',
                                         'running_pods', 'total_pods']].values
            next_state = training_data.iloc[i + 1][['cpu_utilization', 'memory_utilization',
                                                   'running_pods', 'total_pods']].values
            
            # Calculate reward based on resource utilization and cost
            reward = (state[0] + state[1]) / 2.0  # Average utilization
            reward -= abs(0.8 - reward) * 0.5  # Penalty for deviation from target utilization
            
            # Store experience
            action = self.dqn_agent.act(state)
            self.dqn_agent.remember(state, action, reward, next_state, False)
            
            # Train on batch
            if len(self.dqn_agent.memory) >= self.batch_size:
                self.dqn_agent.replay(self.batch_size)
                
    def optimize_resources(self, deployment_name: str, namespace: str) -> Dict:
        """Run optimization cycle"""
        # Get current state
        current_state = self.k8s_manager.get_cluster_state()
        
        # Predict future resource demands
        future_demands = self.forecasting_model.predict(steps=12)  # Next hour
        
        # Use DQN to determine high-level action
        state = np.concatenate([
            current_state,
            [np.mean(future_demands), np.std(future_demands)]
        ])
        action = self.dqn_agent.act(state)
        
        # Use GA to fine-tune specific resource allocations
        best_chromosome, _ = self.genetic_algorithm.evolve(generations=20)
        
        # Extract optimization results
        cpu_request = f"{best_chromosome[0]}m"
        memory_request = f"{best_chromosome[1]}Mi"
        replicas = int(best_chromosome[2])
        
        # Apply changes
        self.k8s_manager.update_resource_requests(
            name=deployment_name,
            namespace=namespace,
            cpu_request=cpu_request,
            memory_request=memory_request
        )
        self.k8s_manager.scale_deployment(
            name=deployment_name,
            namespace=namespace,
            replicas=replicas
        )
        
        return {
            'cpu_request': cpu_request,
            'memory_request': memory_request,
            'replicas': replicas,
            'predicted_utilization': np.mean(future_demands)
        }
        
    def run_optimization_loop(self, deployment_name: str, namespace: str) -> None:
        """Main optimization loop"""
        while True:
            try:
                # Collect metrics
                metrics_df = self.collect_metrics_history(hours=24)
                
                # Retrain models periodically
                self.train_models(metrics_df)
                
                # Run optimization
                results = self.optimize_resources(deployment_name, namespace)
                print(f"Optimization results: {results}")
                
                # Wait for next update interval
                time.sleep(self.update_interval * 60)
                
            except Exception as e:
                print(f"Error in optimization loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
                
if __name__ == "__main__":
    import time
    
    # Initialize optimizer
    optimizer = ResourceOptimizer()
    
    # Get deployment details from environment variables
    deployment_name = os.getenv("DEPLOYMENT_NAME", "default-deployment")
    namespace = os.getenv("NAMESPACE", "default")
    
    # Run optimization loop
    optimizer.run_optimization_loop(deployment_name, namespace) 