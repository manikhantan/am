import pandas as pd
from typing import Dict, List, Any
from .llm_service import LLMService

class AnalysisEngine:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    def generate_analysis_plan(self, data: pd.DataFrame, user_prompt: str) -> Dict:
        """Generate a comprehensive analysis plan using LLM"""
        
        # Prepare data information for LLM
        data_info = self._prepare_data_info(data)
        
        # Generate plan using LLM
        plan = self.llm_service.generate_analysis_plan(data_info, user_prompt)
        
        # Validate and enhance the plan
        enhanced_plan = self._enhance_plan(plan, data)
        
        return enhanced_plan
    
    def _prepare_data_info(self, data: pd.DataFrame) -> Dict:
        """Prepare comprehensive data information for analysis planning"""
        
        # Basic information
        info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'sample_data': data.head(3).to_dict()
        }
        
        # Statistical information for numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info['numeric_stats'] = data[numeric_cols].describe().to_dict()
        
        # Categorical information
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['categorical_info'] = {}
            for col in categorical_cols:
                info['categorical_info'][col] = {
                    'unique_values': data[col].nunique(),
                    'top_values': data[col].value_counts().head(5).to_dict()
                }
        
        # Data quality assessment
        info['data_quality'] = {
            'total_missing': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
            'duplicate_rows': data.duplicated().sum(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }
        
        return info
    
    def _enhance_plan(self, plan: Dict, data: pd.DataFrame) -> Dict:
        """Enhance the generated plan with additional metadata and validation"""
        
        # Add unique IDs to steps if not present
        for i, step in enumerate(plan.get('steps', [])):
            if 'id' not in step:
                step['id'] = f"step_{i+1}"
        
        # Validate dependencies
        step_ids = [step['id'] for step in plan.get('steps', [])]
        for step in plan.get('steps', []):
            # Remove invalid dependencies
            step['dependencies'] = [dep for dep in step.get('dependencies', []) if dep in step_ids]
        
        # Add execution metadata
        plan['metadata'] = {
            'total_steps': len(plan.get('steps', [])),
            'estimated_total_time': self._estimate_total_time(plan.get('steps', [])),
            'data_size': data.shape[0],
            'complexity_score': self._calculate_complexity_score(plan.get('steps', []), data),
            'parallel_groups': self._identify_parallel_groups(plan.get('steps', []))
        }
        
        return plan
    
    def _estimate_total_time(self, steps: List[Dict]) -> str:
        """Estimate total execution time for the analysis plan"""
        
        time_mapping = {
            'seconds': 1,
            'minute': 60,
            'minutes': 60,
            'hour': 3600,
            'hours': 3600
        }
        
        total_seconds = 0
        for step in steps:
            time_str = step.get('estimated_time', '1 minute').lower()
            
            # Parse time string
            parts = time_str.split()
            if len(parts) >= 2:
                try:
                    value = float(parts[0])
                    unit = parts[1].rstrip('s')  # Remove plural 's'
                    total_seconds += value * time_mapping.get(unit, 60)
                except ValueError:
                    total_seconds += 60  # Default to 1 minute
        
        # Convert back to human readable format
        if total_seconds < 60:
            return f"{int(total_seconds)} seconds"
        elif total_seconds < 3600:
            return f"{int(total_seconds/60)} minutes"
        else:
            return f"{int(total_seconds/3600)} hours {int((total_seconds%3600)/60)} minutes"
    
    def _calculate_complexity_score(self, steps: List[Dict], data: pd.DataFrame) -> int:
        """Calculate complexity score for the analysis plan (1-10)"""
        
        score = 0
        
        # Data size factor
        if data.shape[0] > 100000:
            score += 3
        elif data.shape[0] > 10000:
            score += 2
        else:
            score += 1
        
        # Number of steps factor
        if len(steps) > 10:
            score += 3
        elif len(steps) > 5:
            score += 2
        else:
            score += 1
        
        # Analysis type complexity
        analysis_types = [step.get('analysis_type', 'exploratory') for step in steps]
        if 'modeling' in analysis_types:
            score += 3
        elif 'statistical' in analysis_types:
            score += 2
        else:
            score += 1
        
        # Dependencies complexity
        total_deps = sum(len(step.get('dependencies', [])) for step in steps)
        if total_deps > 5:
            score += 1
        
        return min(score, 10)
    
    def _identify_parallel_groups(self, steps: List[Dict]) -> List[List[str]]:
        """Identify which steps can be executed in parallel"""
        
        from utils.dependency_resolver import DependencyResolver
        resolver = DependencyResolver()
        
        try:
            execution_groups = resolver.resolve_dependencies(steps)
            return [[step['id'] for step in group] for group in execution_groups]
        except Exception:
            # Fallback to sequential execution
            return [[step['id']] for step in steps]
