from flask import Flask, render_template, request, jsonify
import pandas as pd
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional
import json
from functools import lru_cache

app = Flask(__name__)

# Initialize the retriever
API_KEY = 'YOUR_GEMINI_API_KEY'  # Replace with your actual API key
DATA_PATH = 'recipe_data.pkl'

class RecipeRetriever:
    def __init__(self, api_key: str, data_path: str):
        """
        Initialize the recipe retriever with API key and data path.
        
        Args:
            api_key: Google API key for Gemini Pro
            data_path: Path to the saved recipe DataFrame
        """
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Load and cache data
        self.df = pd.read_pickle(data_path)
        self.column_info = self._get_column_info()
        
    def _get_column_info(self) -> str:
        """Generate column information description."""
        columns = self.df.columns.tolist()
        sample_values = {col: self.df[col].iloc[0] for col in columns}
        
        return f"""DataFrame columns and sample values:
        {json.dumps({col: str(val)[:100] + '...' if len(str(val)) > 100 else str(val) 
                    for col, val in sample_values.items()}, indent=2)}"""

    @lru_cache(maxsize=128)
    def _get_filter_query(self, user_query: str) -> str:
        """
        Get filter query from LLM based on user question.
        Uses caching to avoid repeated API calls for similar questions.
        """
        prompt = f"""Based on this user question: "{user_query}"
        And these available columns: {self.column_info}
        
        Generate a Python dictionary with filtering conditions that would help answer the question.
        Use only exact match conditions or 'contains' string operations.
        
        Return ONLY a Python dictionary like this example:
        {{'column_name': 'value'}} for exact match
        or {{'column_name': {{'contains': 'value'}}}} for partial match
        
        If no relevant filters apply, return an empty dictionary {{}}.
        
        RESPOND WITH ONLY THE DICTIONARY, NO OTHER TEXT."""
        
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _apply_filters(self, filter_dict: Dict) -> pd.DataFrame:
        """Apply filters to DataFrame based on filter dictionary."""
        filtered_df = self.df.copy()
        
        if not filter_dict:
            return filtered_df.head(5)  # Return top 5 if no filters
            
        for col, condition in filter_dict.items():
            if col not in filtered_df.columns:
                continue
                
            if isinstance(condition, dict) and 'contains' in condition:
                filtered_df = filtered_df[filtered_df[col].str.contains(condition['contains'], 
                                                                      case=False, 
                                                                      na=False)]
            else:
                filtered_df = filtered_df[filtered_df[col] == condition]
        
        return filtered_df.head(5)  # Limit to top 5 matches

    def _get_answer(self, user_query: str, filtered_data: pd.DataFrame) -> str:
        """Generate answer based on filtered data and user query."""
        if filtered_data.empty:
            return "I couldn't find any recipes matching your criteria."
        
        data_str = filtered_data.to_string()
        prompt = f"""Question: {user_query}

        Available recipe data:
        {data_str}

        Please provide a helpful answer based on this data. If the data doesn't contain enough information to answer the question properly, please say so.
        Focus on being accurate and concise. Include specific details from the recipes when relevant."""
        
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def query(self, user_query: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Process user query and return answer with optional filtered DataFrame.
        
        Args:
            user_query: User's question about recipes
            
        Returns:
            Tuple of (answer string, filtered DataFrame or None)
        """
        try:
            # Get and parse filter conditions
            filter_str = self._get_filter_query(user_query)
            filter_dict = eval(filter_str)  # Convert string to dictionary
            
            # Apply filters
            filtered_df = self._apply_filters(filter_dict)
            
            # Generate answer
            answer = self._get_answer(user_query, filtered_df)
            
            return answer, filtered_df
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}", None

retriever = RecipeRetriever(API_KEY, DATA_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_question = request.json.get('question', '')
    
    if not user_question:
        return jsonify({'error': 'No question provided'})
    
    try:
        answer, filtered_df = retriever.query(user_question)
        
        # Convert filtered DataFrame to HTML table with styling
        table_html = ''
        if filtered_df is not None and not filtered_df.empty:
            table_html = filtered_df.to_html(classes=['table', 'table-striped', 'table-hover'],
                                           index=False,
                                           escape=False)
        
        return jsonify({
            'answer': answer,
            'table': table_html
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)