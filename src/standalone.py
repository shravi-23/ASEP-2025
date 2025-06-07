import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from forecasting.hybrid_model import HybridForecastingModel
from optimization.dqn_agent import DQNAgent
from optimization.genetic_algorithm import GeneticAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandaloneOptimizer:
    def __init__(
        self,
        forecast_sequence_length: int = 10,
        dqn_state_size: int = 6,
        dqn_action_size: int = 5,
        ga_population_size: int = 50,
        ga_chromosome_length: int = 3
    ):
        try:
            # Initialize components
            self.forecasting_model = HybridForecastingModel(
                sequence_length=forecast_sequence_length
            )
            self.dqn_agent = DQNAgent(
                state_size=dqn_state_size,
                action_size=dqn_action_size
            )
            
            # Initialize GA with a simplified fitness function
            def fitness_function(chromosome: np.ndarray) -> float:
                cpu_allocation = chromosome[0]
                memory_allocation = chromosome[1]
                replicas = chromosome[2]
                
                # Simplified cost and performance metrics
                cost = (cpu_allocation * memory_allocation * replicas) / 100.0
                target_utilization = 0.8
                current_utilization = 0.6 + np.random.normal(0, 0.1)
                performance = 1 - abs(target_utilization - current_utilization)
                
                return 0.7 * performance - 0.3 * cost
                
            self.genetic_algorithm = GeneticAlgorithm(
                population_size=ga_population_size,
                chromosome_length=ga_chromosome_length,
                fitness_func=fitness_function
            )
            
            self.batch_size = 32
            self.update_interval = 1  # Changed to 1 minute for demo
            logger.info("Successfully initialized StandaloneOptimizer")
            
        except Exception as e:
            logger.error(f"Error initializing StandaloneOptimizer: {str(e)}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and validate input data"""
        try:
            # Ensure required columns exist
            required_columns = ['timestamp', 'cpu_utilization', 'memory_utilization', 'running_pods', 'total_pods']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Convert timestamp to datetime if it's not already
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Ensure values are within valid ranges
            if not (0 <= df['cpu_utilization'].min() <= df['cpu_utilization'].max() <= 1):
                raise ValueError("CPU utilization must be between 0 and 1")
            if not (0 <= df['memory_utilization'].min() <= df['memory_utilization'].max() <= 1):
                raise ValueError("Memory utilization must be between 0 and 1")
            if not (df['running_pods'] <= df['total_pods']).all():
                raise ValueError("Running pods cannot exceed total pods")
                
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
            
    def train_models(self, training_data: pd.DataFrame) -> None:
        """Train both forecasting and optimization models"""
        try:
            # Preprocess data
            training_data = self.preprocess_data(training_data)
            
            logger.info("Training Hybrid Forecasting Model (LSTM + ARIMA)...")
            self.forecasting_model.fit(
                training_data['cpu_utilization'],
                arima_order=(2, 1, 2)
            )
            
            logger.info("Training DQN Agent...")
            for i in range(len(training_data) - 1):
                base_state = np.array([
                    training_data.iloc[i]['cpu_utilization'],
                    training_data.iloc[i]['memory_utilization'],
                    training_data.iloc[i]['running_pods'],
                    training_data.iloc[i]['total_pods']
                ], dtype=np.float32)
                
                future_window = training_data.iloc[i:i+6]['cpu_utilization'].values
                state = np.concatenate([
                    base_state,
                    [np.mean(future_window), np.std(future_window)]
                ]).astype(np.float32)
                
                base_next_state = np.array([
                    training_data.iloc[i + 1]['cpu_utilization'],
                    training_data.iloc[i + 1]['memory_utilization'],
                    training_data.iloc[i + 1]['running_pods'],
                    training_data.iloc[i + 1]['total_pods']
                ], dtype=np.float32)
                
                future_window_next = training_data.iloc[i+1:i+7]['cpu_utilization'].values
                next_state = np.concatenate([
                    base_next_state,
                    [np.mean(future_window_next), np.std(future_window_next)]
                ]).astype(np.float32)
                
                reward = (state[0] + state[1]) / 2.0
                reward -= abs(0.8 - reward) * 0.5
                
                action = self.dqn_agent.act(state)
                self.dqn_agent.remember(state, action, reward, next_state, False)
                
                if len(self.dqn_agent.memory) >= self.batch_size:
                    self.dqn_agent.replay(self.batch_size)
                    
            logger.info("Successfully completed model training")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
            
    def optimize_resources(self) -> Dict:
        """Run optimization cycle"""
        try:
            logger.info("Starting optimization cycle")
            
            # Generate current state
            current_state = np.array([0.6, 0.55, 8, 12], dtype=np.float32)
            
            # Predict future resource demands
            future_demands = self.forecasting_model.predict(steps=12)
            logger.info(f"Predicted Resource Utilization: {future_demands.mean():.2%}")
            
            # Use DQN to determine high-level action
            state = np.concatenate([
                current_state,
                np.array([np.mean(future_demands), np.std(future_demands)], dtype=np.float32)
            ])
            action = self.dqn_agent.act(state)
            logger.info(f"DQN Action Selected: {action}")
            
            # Use GA to fine-tune specific resource allocations
            best_chromosome, fitness = self.genetic_algorithm.evolve(generations=20)
            logger.info(f"GA Optimization Complete (Fitness: {fitness:.3f})")
            
            # Extract optimization results
            cpu_request = f"{int(best_chromosome[0])}m"
            memory_request = f"{int(best_chromosome[1])}Mi"
            replicas = int(best_chromosome[2])
            
            results = {
                'cpu_request': cpu_request,
                'memory_request': memory_request,
                'replicas': replicas,
                'predicted_utilization': float(np.mean(future_demands))
            }
            
            logger.info(f"Optimization Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in resource optimization: {str(e)}")
            raise
            
    def run_optimization_loop(self) -> None:
        """Main optimization loop"""
        try:
            logger.info("Starting continuous optimization loop")
            iteration = 1
            
            while True:
                logger.info(f"Starting iteration {iteration}")
                results = self.optimize_resources()
                
                logger.info(f"Waiting {self.update_interval} minute(s) before next optimization...")
                time.sleep(self.update_interval * 60)
                iteration += 1
                
        except KeyboardInterrupt:
            logger.info("Optimization loop stopped by user")
        except Exception as e:
            logger.error(f"Error in optimization loop: {str(e)}")
            raise
        
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = StandaloneOptimizer()
    
    # Run optimization loop
    optimizer.run_optimization_loop() 