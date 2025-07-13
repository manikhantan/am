import json
from typing import Dict, List, Any, Optional

class PlanParser:
    """Parse and validate analysis plans from LLM responses"""
    
    def __init__(self):
        self.required_step_fields = ['id', 'title', 'description', 'analysis_type']
        self.optional_step_fields = ['dependencies', 'estimated_time', 'code_preview']
    
    def parse_plan(self, plan_data: Any) -> Optional[Dict]:
        """Parse and validate an analysis plan"""
        
        try:
            # Convert to dict if it's a string
            if isinstance(plan_data, str):
                plan_data = json.loads(plan_data)
            
            # Validate plan structure
            if not isinstance(plan_data, dict):
                raise ValueError("Plan must be a dictionary")
            
            # Check required fields
            if 'steps' not in plan_data:
                raise ValueError("Plan must contain 'steps' field")
            
            # Validate steps
            validated_steps = []
            for i, step in enumerate(plan_data['steps']):
                validated_step = self._validate_step(step, i)
                if validated_step:
                    validated_steps.append(validated_step)
            
            # Create validated plan
            validated_plan = {
                'plan_summary': plan_data.get('plan_summary', 'Data analysis plan'),
                'steps': validated_steps,
                'total_steps': len(validated_steps)
            }
            
            return validated_plan
            
        except Exception as e:
            print(f"Error parsing plan: {str(e)}")
            return None
    
    def _validate_step(self, step: Dict, index: int) -> Optional[Dict]:
        """Validate a single analysis step"""
        
        try:
            # Check if step is a dictionary
            if not isinstance(step, dict):
                return None
            
            # Create validated step with required fields
            validated_step = {}
            
            # Add ID if not present
            validated_step['id'] = step.get('id', f'step_{index + 1}')
            
            # Add required fields
            validated_step['title'] = step.get('title', f'Analysis Step {index + 1}')
            validated_step['description'] = step.get('description', 'No description provided')
            validated_step['analysis_type'] = step.get('analysis_type', 'exploratory')
            
            # Add optional fields with defaults
            validated_step['dependencies'] = step.get('dependencies', [])
            validated_step['estimated_time'] = step.get('estimated_time', '2 minutes')
            validated_step['code_preview'] = step.get('code_preview', '')
            
            # Validate dependencies
            if not isinstance(validated_step['dependencies'], list):
                validated_step['dependencies'] = []
            
            # Validate analysis type
            valid_types = ['exploratory', 'statistical', 'visualization', 'modeling', 'cleaning']
            if validated_step['analysis_type'] not in valid_types:
                validated_step['analysis_type'] = 'exploratory'
            
            return validated_step
            
        except Exception as e:
            print(f"Error validating step {index}: {str(e)}")
            return None
    
    def validate_dependencies(self, steps: List[Dict]) -> List[Dict]:
        """Validate and fix step dependencies"""
        
        step_ids = [step['id'] for step in steps]
        
        for step in steps:
            # Remove invalid dependencies
            valid_deps = []
            for dep in step.get('dependencies', []):
                if dep in step_ids and dep != step['id']:
                    valid_deps.append(dep)
            
            step['dependencies'] = valid_deps
        
        return steps
    
    def detect_circular_dependencies(self, steps: List[Dict]) -> List[str]:
        """Detect circular dependencies in the plan"""
        
        def has_circular_dependency(step_id: str, visited: set, path: set) -> bool:
            if step_id in path:
                return True
            
            if step_id in visited:
                return False
            
            visited.add(step_id)
            path.add(step_id)
            
            # Find step with this ID
            step = next((s for s in steps if s['id'] == step_id), None)
            if step:
                for dep in step.get('dependencies', []):
                    if has_circular_dependency(dep, visited, path):
                        return True
            
            path.remove(step_id)
            return False
        
        visited = set()
        circular_deps = []
        
        for step in steps:
            step_id = step['id']
            if step_id not in visited:
                if has_circular_dependency(step_id, visited, set()):
                    circular_deps.append(step_id)
        
        return circular_deps
    
    def fix_plan_issues(self, plan: Dict) -> Dict:
        """Fix common issues in analysis plans"""
        
        try:
            # Fix missing or invalid fields
            if 'steps' not in plan:
                plan['steps'] = []
            
            # Validate and fix steps
            plan['steps'] = self.validate_dependencies(plan['steps'])
            
            # Check for circular dependencies
            circular_deps = self.detect_circular_dependencies(plan['steps'])
            if circular_deps:
                # Remove circular dependencies
                for step in plan['steps']:
                    if step['id'] in circular_deps:
                        step['dependencies'] = []
            
            # Ensure plan summary exists
            if 'plan_summary' not in plan:
                plan['plan_summary'] = 'Automated data analysis plan'
            
            # Add metadata if not present
            if 'total_steps' not in plan:
                plan['total_steps'] = len(plan['steps'])
            
            return plan
            
        except Exception as e:
            print(f"Error fixing plan issues: {str(e)}")
            return plan
    
    def create_fallback_plan(self, data_columns: List[str]) -> Dict:
        """Create a fallback analysis plan when LLM fails"""
        
        fallback_steps = [
            {
                'id': 'step_1',
                'title': 'Data Overview',
                'description': 'Basic data exploration including shape, types, and missing values',
                'analysis_type': 'exploratory',
                'dependencies': [],
                'estimated_time': '1 minute',
                'code_preview': 'df.info(), df.describe()'
            },
            {
                'id': 'step_2',
                'title': 'Missing Value Analysis',
                'description': 'Analyze patterns in missing data',
                'analysis_type': 'exploratory',
                'dependencies': ['step_1'],
                'estimated_time': '2 minutes',
                'code_preview': 'df.isnull().sum()'
            },
            {
                'id': 'step_3',
                'title': 'Descriptive Statistics',
                'description': 'Calculate summary statistics for numerical columns',
                'analysis_type': 'statistical',
                'dependencies': ['step_1'],
                'estimated_time': '2 minutes',
                'code_preview': 'df.select_dtypes(include=[np.number]).describe()'
            },
            {
                'id': 'step_4',
                'title': 'Data Visualization',
                'description': 'Create basic visualizations for key variables',
                'analysis_type': 'visualization',
                'dependencies': ['step_2', 'step_3'],
                'estimated_time': '3 minutes',
                'code_preview': 'plt.figure(); df.hist()'
            }
        ]
        
        return {
            'plan_summary': 'Fallback analysis plan for basic data exploration',
            'steps': fallback_steps,
            'total_steps': len(fallback_steps)
        }
