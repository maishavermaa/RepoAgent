#!/usr/bin/env python3

import os
import ssl
import urllib3
import threading
import time
import requests
from dotenv import load_dotenv

# Disable SSL verification warnings and ChromaDB telemetry
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONHTTPSVERIFY'] = '0'

# DISABLE CHROMADB TELEMETRY to avoid SSL blocking
os.environ['CHROMA_TELEMETRY'] = 'false'
os.environ['ANONYMIZED_TELEMETRY'] = 'false'

# Load environment variables
load_dotenv()

from code_indexer import CodeIndexer
import mcp_server

# Helper functions to call the MCP server API endpoints
def call_search_code(query: str, max_results: int = 5) -> str:
    """Call the search_code function via MCP server API"""
    try:
        response = requests.post("http://127.0.0.1:8000/search_code", 
                               json={"query": query, "max_results": max_results})
        return response.json()["result"]
    except Exception as e:
        return f"Error calling search_code: {str(e)}"

def call_list_project_structure() -> str:
    """Call the list_project_structure function via MCP server API"""
    try:
        response = requests.post("http://127.0.0.1:8000/list_project_structure")
        return response.json()["result"]
    except Exception as e:
        return f"Error calling list_project_structure: {str(e)}"

def call_get_file_content(file_path: str) -> str:
    """Call the get_file_content function via MCP server API"""
    try:
        response = requests.post("http://127.0.0.1:8000/get_file_content", 
                               json={"file_path": file_path})
        return response.json()["result"]
    except Exception as e:
        return f"Error calling get_file_content: {str(e)}"

def call_explain_function(function_name: str) -> str:
    """Call the explain_function function via MCP server API"""
    try:
        response = requests.post("http://127.0.0.1:8000/explain_function", 
                               json={"function_name": function_name})
        return response.json()["result"]
    except Exception as e:
        return f"Error calling explain_function: {str(e)}"

def main():
    print("🚀 Code Analysis Chatbot (Direct MCP)")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return
    print("✅ API key loaded")
    
    # Connect to existing data (no re-indexing)
    print("🔍 Connecting to existing data...")
    try:
        indexer = CodeIndexer()
        count = indexer.collection.count()
        print(f"📊 Found {count} file summaries in database")
        
        if count == 0:
            print("❌ No data found. Run indexing first.")
            return
        
        # Set the global instance for MCP server
        import code_indexer
        code_indexer._CODE_INDEXER_INSTANCE = indexer
        
        print("🌐 Starting MCP server...")
        server_thread = threading.Thread(
            target=mcp_server.start_server,
            args=("127.0.0.1", 8000),
            daemon=True
        )
        server_thread.start()
        time.sleep(3)
        
        print("\n" + "="*50)
        print("🎉 CHATBOT READY!")
        print(f"📊 {count} file summaries loaded")
        print("💡 File-summary based indexing active!")
        print("🌐 MCP server running on http://127.0.0.1:8000")
        print("="*50)
        
        # Direct function testing via API calls
        print("\n🧪 Testing MCP functions via API...")
        
        # Test search
        print("\n🔍 Testing search functionality:")
        results = indexer.search("python class", 3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['file_path']}")
            print(f"     📝 {result['file_type']} | {result['language']} | {result['line_count']} lines")
            print(f"     🎯 {result['purpose'][:80]}...")
        
        # Test project structure via API
        print(f"\n📁 Testing project structure via API...")
        structure = call_list_project_structure()
        print(structure[:500] + "..." if len(structure) > 500 else structure)
        
        # Interactive mode with API calls
        print("\n" + "="*50)
        print("🎮 INTERACTIVE MODE")
        print("Available commands:")
        print("1. search <query>     - Search for files")
        print("2. structure          - Show project structure") 
        print("3. file <path>        - Get file content")
        print("4. explain <function> - Explain a function")
        print("5. exit               - Quit")
        print("="*50)
        
        while True:
            try:
                command = input("\n🤔 Command: ").strip()
                
                if command.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye!")
                    break
                
                if not command:
                    continue
                
                parts = command.split(" ", 1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if cmd == "search":
                    if not arg:
                        print("❌ Usage: search <query>")
                        continue
                    print(f"\n🔍 Searching for: '{arg}'")
                    results = indexer.search(arg, 5)
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {result['file_path']}")
                        print(f"   📝 {result['file_type']} | {result['language']} | {result['line_count']} lines")
                        print(f"   🎯 {result['purpose']}")
                
                elif cmd == "structure":
                    print("\n📁 Project Structure:")
                    structure = call_list_project_structure()
                    print(structure)
                
                elif cmd == "file":
                    if not arg:
                        print("❌ Usage: file <path>")
                        continue
                    print(f"\n📄 File content for: {arg}")
                    content = call_get_file_content(arg)
                    print(content[:1000] + "..." if len(content) > 1000 else content)
                
                elif cmd == "explain":
                    if not arg:
                        print("❌ Usage: explain <function_name>")
                        continue
                    print(f"\n🔍 Explaining function: {arg}")
                    explanation = call_explain_function(arg)
                    print(explanation)
                
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("Available: search, structure, file, explain, exit")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 