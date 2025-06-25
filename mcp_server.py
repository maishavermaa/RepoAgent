import os
import ast
import json
import re
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from agents import function_tool

from code_indexer import CodeIndexer, get_indexed_codebase

app = FastAPI()
indexer = None

class SearchParams(BaseModel):
    query: str
    max_results: int = 5

class FunctionParams(BaseModel):
    function_name: str

class FileContentParams(BaseModel):
    file_path: str

class ClassParams(BaseModel):
    class_name: str

@function_tool
def search_code(query: str, max_results: int = 5) -> str:
    """
    Search the codebase for relevant file summaries based on the query.
    
    Args:
        query: The search query to find relevant files
        max_results: Maximum number of results to return
        
    Returns:
        Formatted file summaries with metadata and descriptions
    """
    print(f"FUNCTION_TOOL: search_code called with query: '{query}', max_results: {max_results}")
    global indexer
    if not indexer:
        indexer = get_indexed_codebase()
        if not indexer:
            return "No codebase has been indexed yet. Please upload a repository first."
    
    results = indexer.search(query, max_results)
    
    if not results:
        return "No relevant files found for the query."
    
    formatted_results = "Found relevant files:\n\n"
    for result in results:
        formatted_results += f"📄 {result['file_path']}\n"
        formatted_results += f"Language: {result['language']}, Type: {result['file_type']}\n"
        formatted_results += f"Lines: {result['line_count']}, Complexity: {result['complexity_score']}\n"
        formatted_results += f"Purpose: {result['purpose']}\n"
    
    return formatted_results

@function_tool
def explain_function(function_name: str) -> str:
    """
    Explain a function from the codebase, including its purpose, parameters, and return type.
    
    Args:
        function_name: The name of the function to explain
        
    Returns:
        Explanation of the function including its signature, docstring, and implementation
    """
    print(f"FUNCTION_TOOL: explain_function called with function_name: '{function_name}'")
    global indexer
    if not indexer:
        indexer = get_indexed_codebase()
        if not indexer:
            return "No codebase has been indexed yet. Please upload a repository first."
    
    # Search for the function in the codebase
    function_results = indexer.search(f"def {function_name}", 10)
    
    if not function_results:
        return f"Function '{function_name}' not found in the codebase."
    
    # Try to find an exact match for the function
    function_info = None
    for result in function_results:
        try:
            tree = ast.parse(result['content'])
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    function_info = {
                        'file_path': result['file_path'],
                        'content': result['content'],
                        'node': node,
                        'language': result['language'],
                        'start_line': node.lineno,
                        'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                    }
                    break
            if function_info:
                break
        except SyntaxError:
            # Skip files that can't be parsed as Python
            continue
    
    if not function_info:
        # If no exact match is found, return the closest matches
        explanation = f"Couldn't find an exact match for function '{function_name}', but here are some potential matches:\n\n"
        for result in function_results[:3]:
            explanation += f"File: {result['file_path']}\n"
            explanation += f"```{result['language']}\n{result['content']}\n```\n\n"
        return explanation
    
    # Extract function signature, docstring, and code
    node = function_info['node']
    
    # Get function signature
    args_str = []
    for arg in node.args.args:
        arg_name = arg.arg
        
        # Try to get type annotation if available
        if hasattr(arg, 'annotation') and arg.annotation:
            try:
                arg_type = ast.unparse(arg.annotation)
                args_str.append(f"{arg_name}: {arg_type}")
            except:
                args_str.append(arg_name)
        else:
            args_str.append(arg_name)
    
    # Handle return type annotation
    return_type = "Any"
    if hasattr(node, 'returns') and node.returns:
        try:
            return_type = ast.unparse(node.returns)
        except:
            pass
    
    # Get function docstring
    docstring = ast.get_docstring(node) or "No docstring available"
    
    # Get function code
    lines = function_info['content'].split('\n')
    function_code = '\n'.join(lines[node.lineno-1:node.end_lineno])
    
    explanation = f"## Function: {function_name}\n\n"
    explanation += f"**File:** {function_info['file_path']}\n\n"
    explanation += f"**Signature:** def {function_name}({', '.join(args_str)}) -> {return_type}\n\n"
    explanation += f"**Docstring:**\n{docstring}\n\n"
    explanation += f"**Code:**\n```{function_info['language']}\n{function_code}\n```\n\n"
    
    return explanation

@function_tool
def get_file_content(file_path: str) -> str:
    """
    Get the content of a specific file from the codebase.
    
    Args:
        file_path: The path of the file to retrieve
        
    Returns:
        The content of the specified file
    """
    print(f"FUNCTION_TOOL: get_file_content called with file_path: '{file_path}'")
    global indexer
    if not indexer:
        indexer = get_indexed_codebase()
        if not indexer:
            return "No codebase has been indexed yet. Please upload a repository first."
    
    file_content = indexer.get_file_content(file_path)
    
    if not file_content:
        # Try to find a partial match
        all_files = indexer.get_all_files()
        potential_matches = [f for f in all_files if file_path.lower() in f.lower()]
        
        if potential_matches:
            response = f"File '{file_path}' not found exactly. Did you mean one of these?\n\n"
            for match in potential_matches[:5]:
                response += f"- {match}\n"
            return response
        else:
            return f"File '{file_path}' not found in the codebase."
    
    file_extension = os.path.splitext(file_path)[1].lstrip('.')
    return f"Content of {file_path}:\n\n```{file_extension}\n{file_content}\n```"

@function_tool
def list_project_structure() -> str:
    """
    List the directory structure of the project.
    
    Returns:
        A formatted tree representation of the project structure
    """
    print(f"FUNCTION_TOOL: list_project_structure called")
    global indexer
    if not indexer:
        indexer = get_indexed_codebase()
        if not indexer:
            return "No codebase has been indexed yet. Please upload a repository first."
    
    all_files = indexer.get_all_files()
    
    if not all_files:
        return "No files found in the codebase."
    
    # Create a nested dictionary representing the directory structure
    tree = {}
    for file_path in all_files:
        parts = file_path.split('/')
        current = tree
        for part in parts[:-1]:  # Process directories
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Add file
        if '__files__' not in current:
            current['__files__'] = []
        current['__files__'].append(parts[-1])
    
    # Format the tree
    def format_tree(node, prefix='', is_last=True, is_root=False):
        result = ''
        if not is_root:
            result = prefix + ('└── ' if is_last else '├── ') + (list(node.keys())[0] if isinstance(node, dict) and '__files__' not in node else '') + '\n'
            prefix = prefix + ('    ' if is_last else '│   ')
        
        if isinstance(node, dict):
            # Process directories
            keys = [k for k in node.keys() if k != '__files__']
            for i, key in enumerate(keys):
                is_last_child = (i == len(keys) - 1 and '__files__' not in node)
                result += format_tree({key: node[key]}, prefix, is_last_child)
            
            # Process files
            if '__files__' in node:
                files = node['__files__']
                for i, file in enumerate(files):
                    is_last_file = (i == len(files) - 1)
                    result += prefix + ('└── ' if is_last_file else '├── ') + file + '\n'
        
        return result
    
    tree_representation = "Project Structure:\n\n"
    tree_representation += format_tree(tree, is_root=True)
    
    return tree_representation

@function_tool
def explain_class(class_name: str) -> str:
    """
    Explain a class from the codebase, including its methods, properties, and inheritance.
    
    Args:
        class_name: The name of the class to explain
        
    Returns:
        Explanation of the class including its methods, properties and inheritance
    """
    print(f"FUNCTION_TOOL: explain_class called with class_name: '{class_name}'")
    global indexer
    if not indexer:
        indexer = get_indexed_codebase()
        if not indexer:
            return "No codebase has been indexed yet. Please upload a repository first."
    
    # Search for the class in the codebase
    class_results = indexer.search(f"class {class_name}", 10)
    
    if not class_results:
        return f"Class '{class_name}' not found in the codebase."
    
    # Try to find an exact match for the class
    class_info = None
    for result in class_results:
        try:
            tree = ast.parse(result['content'])
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    class_info = {
                        'file_path': result['file_path'],
                        'content': result['content'],
                        'node': node,
                        'language': result['language'],
                        'start_line': node.lineno,
                        'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                    }
                    break
            if class_info:
                break
        except SyntaxError:
            # Skip files that can't be parsed as Python
            continue
    
    if not class_info:
        # If no exact match is found, return the closest matches
        explanation = f"Couldn't find an exact match for class '{class_name}', but here are some potential matches:\n\n"
        for result in class_results[:3]:
            explanation += f"File: {result['file_path']}\n"
            explanation += f"```{result['language']}\n{result['content']}\n```\n\n"
        return explanation
    
    # Extract class information
    node = class_info['node']
    
    # Get class inheritance
    bases = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except:
            bases.append("Unknown")
    
    # Get class docstring
    docstring = ast.get_docstring(node) or "No docstring available"
    
    # Extract methods and properties
    methods = []
    properties = []
    
    for child in node.body:
        if isinstance(child, ast.FunctionDef):
            methods.append(child.name)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    properties.append(target.id)
    
    # Get class code
    lines = class_info['content'].split('\n')
    class_code = '\n'.join(lines[node.lineno-1:node.end_lineno])
    
    explanation = f"## Class: {class_name}\n\n"
    explanation += f"**File:** {class_info['file_path']}\n\n"
    
    if bases:
        explanation += f"**Inherits from:** {', '.join(bases)}\n\n"
    else:
        explanation += "**Inherits from:** No explicit inheritance\n\n"
    
    explanation += f"**Docstring:**\n{docstring}\n\n"
    
    if methods:
        explanation += f"**Methods:** {', '.join(methods)}\n\n"
    else:
        explanation += "**Methods:** None\n\n"
    
    if properties:
        explanation += f"**Properties:** {', '.join(properties)}\n\n"
    else:
        explanation += "**Properties:** None\n\n"
    
    explanation += f"**Code:**\n```{class_info['language']}\n{class_code}\n```\n\n"
    
    return explanation

@app.post("/search_code")
async def api_search_code(params: SearchParams):
    return {"result": search_code(params.query, params.max_results)}

@app.post("/explain_function")
async def api_explain_function(params: FunctionParams):
    return {"result": explain_function(params.function_name)}

@app.post("/get_file_content")
async def api_get_file_content(params: FileContentParams):
    return {"result": get_file_content(params.file_path)}

@app.post("/list_project_structure")
async def api_list_project_structure():
    return {"result": list_project_structure()}

@app.post("/explain_class")
async def api_explain_class(params: ClassParams):
    return {"result": explain_class(params.class_name)}

@app.get("/")
async def root():
    return {"message": "Codebase Analysis MCP Server is running"}

def start_server(host="127.0.0.1", port=8000):
    """Start the MCP server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server() 