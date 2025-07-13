from typing import List, Dict, Any, Set
from collections import defaultdict, deque

class DependencyResolver:
    """Resolve dependencies between analysis steps for parallel execution"""
    
    def __init__(self):
        pass
    
    def resolve_dependencies(self, steps: List[Dict]) -> List[List[Dict]]:
        """Resolve dependencies and return execution groups for parallel processing"""
        
        try:
            # Build dependency graph
            graph = self._build_dependency_graph(steps)
            
            # Detect circular dependencies
            if self._has_circular_dependencies(graph):
                # Remove circular dependencies
                graph = self._remove_circular_dependencies(graph, steps)
            
            # Perform topological sort to get execution order
            execution_groups = self._topological_sort_groups(graph, steps)
            
            return execution_groups
            
        except Exception as e:
            print(f"Error resolving dependencies: {str(e)}")
            # Fall back to sequential execution
            return [[step] for step in steps]
    
    def _build_dependency_graph(self, steps: List[Dict]) -> Dict[str, List[str]]:
        """Build a dependency graph from analysis steps"""
        
        graph = defaultdict(list)
        step_ids = {step['id'] for step in steps}
        
        for step in steps:
            step_id = step['id']
            dependencies = step.get('dependencies', [])
            
            # Only add valid dependencies
            for dep in dependencies:
                if dep in step_ids and dep != step_id:
                    graph[dep].append(step_id)
        
        return dict(graph)
    
    def _has_circular_dependencies(self, graph: Dict[str, List[str]]) -> bool:
        """Check if the dependency graph has circular dependencies"""
        
        visited = set()
        rec_stack = set()
        
        def dfs(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes
        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def _remove_circular_dependencies(self, graph: Dict[str, List[str]], steps: List[Dict]) -> Dict[str, List[str]]:
        """Remove circular dependencies from the graph"""
        
        # Simple approach: remove all dependencies for nodes involved in cycles
        visited = set()
        rec_stack = set()
        circular_nodes = set()
        
        def find_circular_nodes(node: str):
            if node in rec_stack:
                circular_nodes.add(node)
                return
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                find_circular_nodes(neighbor)
            
            rec_stack.remove(node)
        
        # Find all nodes involved in cycles
        for node in graph:
            if node not in visited:
                find_circular_nodes(node)
        
        # Remove dependencies for circular nodes
        for step in steps:
            if step['id'] in circular_nodes:
                step['dependencies'] = []
        
        # Rebuild graph
        return self._build_dependency_graph(steps)
    
    def _topological_sort_groups(self, graph: Dict[str, List[str]], steps: List[Dict]) -> List[List[Dict]]:
        """Perform topological sort and group steps that can run in parallel"""
        
        # Create step lookup
        step_lookup = {step['id']: step for step in steps}
        
        # Calculate in-degrees
        in_degree = defaultdict(int)
        all_nodes = set()
        
        # Add all step IDs to nodes
        for step in steps:
            all_nodes.add(step['id'])
        
        # Calculate in-degrees
        for node in all_nodes:
            in_degree[node] = 0
        
        for node in graph:
            for neighbor in graph[node]:
                if neighbor in all_nodes:
                    in_degree[neighbor] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        execution_groups = []
        
        while queue:
            # All nodes in current queue level can be executed in parallel
            current_group = []
            next_queue = deque()
            
            while queue:
                node = queue.popleft()
                if node in step_lookup:
                    current_group.append(step_lookup[node])
                
                # Update in-degrees of neighbors
                for neighbor in graph.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            
            if current_group:
                execution_groups.append(current_group)
            
            queue = next_queue
        
        return execution_groups
    
    def validate_execution_groups(self, execution_groups: List[List[Dict]]) -> bool:
        """Validate that execution groups respect dependencies"""
        
        executed_steps = set()
        
        for group in execution_groups:
            # Check that all dependencies of steps in this group have been executed
            for step in group:
                dependencies = step.get('dependencies', [])
                for dep in dependencies:
                    if dep not in executed_steps:
                        return False
            
            # Mark steps in this group as executed
            for step in group:
                executed_steps.add(step['id'])
        
        return True
    
    def get_execution_statistics(self, execution_groups: List[List[Dict]]) -> Dict[str, Any]:
        """Get statistics about the execution plan"""
        
        total_steps = sum(len(group) for group in execution_groups)
        parallel_potential = max(len(group) for group in execution_groups) if execution_groups else 0
        
        return {
            'total_steps': total_steps,
            'execution_groups': len(execution_groups),
            'max_parallel_steps': parallel_potential,
            'parallelization_ratio': parallel_potential / total_steps if total_steps > 0 else 0,
            'sequential_time_saved': total_steps - len(execution_groups) if len(execution_groups) > 0 else 0
        }
    
    def optimize_execution_order(self, execution_groups: List[List[Dict]]) -> List[List[Dict]]:
        """Optimize execution order within groups based on estimated time and complexity"""
        
        optimized_groups = []
        
        for group in execution_groups:
            if len(group) <= 1:
                optimized_groups.append(group)
                continue
            
            # Sort by estimated time and complexity
            sorted_group = sorted(group, key=lambda step: (
                self._parse_estimated_time(step.get('estimated_time', '1 minute')),
                step.get('analysis_type', 'exploratory')
            ))
            
            optimized_groups.append(sorted_group)
        
        return optimized_groups
    
    def _parse_estimated_time(self, time_str: str) -> int:
        """Parse estimated time string to seconds for sorting"""
        
        time_str = time_str.lower()
        
        if 'second' in time_str:
            return int(time_str.split()[0]) if time_str.split()[0].isdigit() else 1
        elif 'minute' in time_str:
            return int(time_str.split()[0]) * 60 if time_str.split()[0].isdigit() else 60
        elif 'hour' in time_str:
            return int(time_str.split()[0]) * 3600 if time_str.split()[0].isdigit() else 3600
        else:
            return 60  # Default to 1 minute
