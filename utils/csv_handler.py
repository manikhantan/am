import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any
import io

class CSVHandler:
    """Handle CSV file operations with validation and optimization"""
    
    def __init__(self):
        self.max_file_size = 100 * 1024 * 1024  # 100MB limit
        self.max_rows = 1000000  # 1M rows limit
    
    def load_csv(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load and validate CSV file"""
        
        try:
            # Check file size
            if uploaded_file.size > self.max_file_size:
                st.error(f"File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds limit ({self.max_file_size / 1024 / 1024:.0f}MB)")
                return None
            
            # Try to detect encoding
            encoding = self._detect_encoding(uploaded_file)
            
            # Read CSV with error handling
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
            except UnicodeDecodeError:
                # Try different encodings
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("Could not decode file. Please ensure it's a valid CSV file.")
                    return None
            
            # Validate dataframe
            if df.empty:
                st.error("The uploaded file is empty.")
                return None
            
            # Check row limit
            if len(df) > self.max_rows:
                st.warning(f"File has {len(df)} rows. Using first {self.max_rows} rows for analysis.")
                df = df.head(self.max_rows)
            
            # Basic data cleaning
            df = self._basic_cleaning(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return None
    
    def _detect_encoding(self, uploaded_file) -> str:
        """Detect file encoding"""
        
        try:
            import chardet
            
            # Read first few bytes to detect encoding
            sample = uploaded_file.read(10000)
            uploaded_file.seek(0)  # Reset file pointer
            
            result = chardet.detect(sample)
            return result['encoding'] or 'utf-8'
            
        except ImportError:
            return 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning"""
        
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            # Clean column names
            df.columns = df.columns.str.strip()  # Remove whitespace
            df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)  # Replace special chars
            df.columns = df.columns.str.replace(r'\s+', '_', regex=True)  # Replace whitespace
            
            # Handle duplicate column names
            df = self._handle_duplicate_columns(df)
            
            return df
            
        except Exception as e:
            st.warning(f"Error during basic cleaning: {str(e)}")
            return df
    
    def _handle_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate column names"""
        
        columns = df.columns.tolist()
        seen = set()
        new_columns = []
        
        for col in columns:
            original_col = col
            counter = 1
            
            while col in seen:
                col = f"{original_col}_{counter}"
                counter += 1
            
            seen.add(col)
            new_columns.append(col)
        
        df.columns = new_columns
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data information"""
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'sample_data': df.head(5).to_dict()
        }
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Add categorical column information
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['categorical_info'] = {}
            for col in categorical_cols:
                info['categorical_info'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(10).to_dict()
                }
        
        return info
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        
        try:
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                if df[col].min() >= 0:
                    if df[col].max() <= 255:
                        df[col] = df[col].astype('uint8')
                    elif df[col].max() <= 65535:
                        df[col] = df[col].astype('uint16')
                    elif df[col].max() <= 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    if df[col].min() >= -128 and df[col].max() <= 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() >= -32768 and df[col].max() <= 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype('int32')
            
            # Optimize object columns to category if appropriate
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
            
            return df
            
        except Exception as e:
            st.warning(f"Error optimizing data types: {str(e)}")
            return df
    
    def create_sample(self, df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
        """Create a representative sample of the data"""
        
        if len(df) <= sample_size:
            return df
        
        # Stratified sampling if there are categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            # Use first categorical column for stratification
            strat_col = categorical_cols[0]
            try:
                sample = df.groupby(strat_col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, sample_size // df[strat_col].nunique())))
                )
                return sample.sample(min(len(sample), sample_size))
            except Exception:
                # Fall back to random sampling
                return df.sample(sample_size)
        else:
            # Random sampling
            return df.sample(sample_size)
