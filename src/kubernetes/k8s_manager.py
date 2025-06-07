from kubernetes import client, config
from typing import Dict, List, Optional
import numpy as np

class KubernetesManager:
    def __init__(self):
        # Load Kubernetes configuration
        try:
            config.load_incluster_config()  # Try in-cluster config first
        except config.ConfigException:
            config.load_kube_config()  # Fall back to local config
            
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        
    def get_node_metrics(self) -> List[Dict]:
        """Get resource metrics for all nodes"""
        nodes = self.v1.list_node().items
        metrics = []
        
        for node in nodes:
            allocatable = node.status.allocatable
            capacity = node.status.capacity
            
            metrics.append({
                'name': node.metadata.name,
                'cpu_allocatable': self._parse_cpu(allocatable['cpu']),
                'memory_allocatable': self._parse_memory(allocatable['memory']),
                'cpu_capacity': self._parse_cpu(capacity['cpu']),
                'memory_capacity': self._parse_memory(capacity['memory'])
            })
            
        return metrics
        
    def get_pod_metrics(self, namespace: Optional[str] = None) -> List[Dict]:
        """Get resource metrics for all pods"""
        if namespace:
            pods = self.v1.list_namespaced_pod(namespace).items
        else:
            pods = self.v1.list_pod_for_all_namespaces().items
            
        metrics = []
        for pod in pods:
            containers = pod.spec.containers
            total_cpu_request = 0
            total_memory_request = 0
            
            for container in containers:
                if container.resources.requests:
                    cpu = container.resources.requests.get('cpu', '0')
                    memory = container.resources.requests.get('memory', '0')
                    total_cpu_request += self._parse_cpu(cpu)
                    total_memory_request += self._parse_memory(memory)
                    
            metrics.append({
                'name': pod.metadata.name,
                'namespace': pod.metadata.namespace,
                'cpu_request': total_cpu_request,
                'memory_request': total_memory_request,
                'status': pod.status.phase
            })
            
        return metrics
        
    def scale_deployment(self, name: str, namespace: str, replicas: int) -> None:
        """Scale a deployment to specified number of replicas"""
        try:
            self.apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace=namespace,
                body={'spec': {'replicas': replicas}}
            )
        except client.ApiException as e:
            print(f"Exception when scaling deployment: {e}")
            
    def update_resource_requests(
        self,
        name: str,
        namespace: str,
        cpu_request: str,
        memory_request: str
    ) -> None:
        """Update resource requests for a deployment"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name, namespace)
            
            for container in deployment.spec.template.spec.containers:
                if not container.resources:
                    container.resources = client.V1ResourceRequirements()
                if not container.resources.requests:
                    container.resources.requests = {}
                    
                container.resources.requests['cpu'] = cpu_request
                container.resources.requests['memory'] = memory_request
                
            self.apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=namespace,
                body=deployment
            )
        except client.ApiException as e:
            print(f"Exception when updating resources: {e}")
            
    def get_cluster_state(self) -> np.ndarray:
        """Get current cluster state as a feature vector"""
        node_metrics = self.get_node_metrics()
        pod_metrics = self.get_pod_metrics()
        
        total_cpu_capacity = sum(m['cpu_capacity'] for m in node_metrics)
        total_memory_capacity = sum(m['memory_capacity'] for m in node_metrics)
        total_cpu_requested = sum(m['cpu_request'] for m in pod_metrics)
        total_memory_requested = sum(m['memory_request'] for m in pod_metrics)
        
        return np.array([
            total_cpu_requested / total_cpu_capacity,  # CPU utilization
            total_memory_requested / total_memory_capacity,  # Memory utilization
            len([p for p in pod_metrics if p['status'] == 'Running']),  # Running pods
            len(pod_metrics)  # Total pods
        ])
        
    @staticmethod
    def _parse_cpu(cpu_str: str) -> float:
        """Parse CPU string to number of cores"""
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000
        return float(cpu_str)
        
    @staticmethod
    def _parse_memory(memory_str: str) -> float:
        """Parse memory string to bytes"""
        units = {'Ki': 2**10, 'Mi': 2**20, 'Gi': 2**30, 'Ti': 2**40}
        for unit, multiplier in units.items():
            if memory_str.endswith(unit):
                return float(memory_str[:-2]) * multiplier
        return float(memory_str) 