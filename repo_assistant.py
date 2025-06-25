# repo_assistant.py

#!/usr/bin/env python3
"""
Repository Code Assistant - File-Summary Based Analysis using OpenAI Agent SDK
"""

import os
import ssl
import urllib3
import json
from typing import List, Dict, Any
import argparse
from dotenv import load_dotenv

from openai import OpenAI
import httpx

# Handle SSL certificate issues and disable ChromaDB telemetry
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CHROMA_TELEMETRY'] = 'false'
os.environ['ANONYMIZED_TELEMETRY'] = 'false'

# Import the file-summary based system
from code_indexer import CodeIndexer, get_indexed_codebase

class FileSummaryAssistant:
    """OpenAI-powered code assistant using file summaries"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        # Initialize httpx client with verify=False for SSL issues
        client = httpx.Client(verify=False)
        self.client = OpenAI(api_key=api_key, http_client=client)
        
        self.model = model
        self.indexer = None
        
    def load_file_summaries(self):
        """Load existing file summaries from ChromaDB"""
        self.indexer = get_indexed_codebase()
        if not self.indexer:
            print("No file summaries found. Please run chatbot.py to ingest a repository first.")
            return False
        
        count = self.indexer.collection.count()
        print(f"Loaded {count} file summaries from ChromaDB.")
        return True
    
    def search_files(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant files using file summaries"""
        if not self.indexer:
            return []
        
        return self.indexer.search(query, max_results)
    
    def query_code(self, query: str) -> str:
        """Query the codebase using file summaries and OpenAI"""
        
        def search_code_tool(search_query: str) -> str:
            """Tool function to search through file summaries"""
            relevant_files = self.search_files(search_query, 10)
            
            if not relevant_files:
                return "No relevant files found for the search query."
            
            # Format file summaries for the LLM
            result = "Found relevant files:\n\n"
            for file_info in relevant_files:
                result += f"--- File: {file_info['file_path']} ---\n"
                result += f"Language: {file_info['language']}, Type: {file_info['file_type']}\n"
                result += f"Lines: {file_info['line_count']}, Complexity: {file_info['complexity_score']}\n"
                result += f"Purpose: {file_info['purpose']}\n"
                result += f"Summary: {file_info['summary']}\n\n"
            
            return result
        
        def get_file_content_tool(file_path: str) -> str:
            """Tool function to get complete file content"""
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return f"Content of {file_path}:\n\n```\n{content}\n```"
                else:
                    return f"File not found: {file_path}"
            except Exception as e:
                return f"Error reading file {file_path}: {str(e)}"
        
        # Define tools for the assistant
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": "Search for relevant files in the codebase using file summaries. Returns file metadata, purpose, and summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "The search query to find relevant files"
                            }
                        },
                        "required": ["search_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_file_content",
                    "description": "Get the complete content of a specific file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            }
        ]
        
        # Create system message for file-summary context
        system_message = """You are an expert code assistant that helps users understand codebases.
        
You have access to a file-summary based indexing system where each file has been analyzed and summarized with:
- File purpose and overview
- Key classes and their methods
- Key functions and their parameters  
- Important imports and dependencies
- File type classification
- Complexity metrics

Use the search_code tool to find relevant files based on queries. This returns file summaries, not code chunks.
Use get_file_content tool when you need to examine specific files in detail.

Provide comprehensive explanations that highlight the purpose, architecture patterns, and how files work together.
Since you work with complete file context rather than fragments, you can provide coherent and contextually aware responses.
"""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        
        print("Querying codebase...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            # Handle tool calls
            while response.choices[0].message.tool_calls:
                messages.append(response.choices[0].message)
                
                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"DEBUG: Assistant called {tool_name} with query: '{arguments.get('search_query', arguments.get('file_path', 'unknown'))}'")
                    
                    if tool_name == "search_code":
                        result = search_code_tool(arguments["search_query"])
                        print(f"DEBUG: Search result length: {len(result)}. Content start: '{result[:100]}...'")
                    elif tool_name == "get_file_content":
                        result = get_file_content_tool(arguments["file_path"])
                        print(f"DEBUG: File content length: {len(result)}")
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })
                
                # Get the next response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error querying codebase: {str(e)}"

def main():
    """Main function for the repository assistant"""
    load_dotenv()
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    
    # Initialize assistant
    assistant = FileSummaryAssistant(api_key)
    
    # Load file summaries
    if not assistant.load_file_summaries():
        print("Please run: python3 chatbot.py --zip <your-repo.zip> to ingest a repository first.")
        return
    
    print("\n--- Repository Assistant Interactive Mode (File-Summary Based) ---")
    print("Type your query or a command. Type 'help' for commands, 'exit' or 'quit' to end.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- Ask any question about the codebase")
                print("- 'stats' - Show file summary statistics")
                print("- 'exit' or 'quit' - End session")
                continue
            
            if user_input.lower() == 'stats':
                count = assistant.indexer.collection.count()
                files = assistant.indexer.get_all_files()
                print(f"\nFile Summary Statistics:")
                print(f"Total files indexed: {count}")
                print(f"Total unique file paths: {len(files)}")
                continue
            
            if not user_input:
                continue
            
            # Query the codebase
            response = assistant.query_code(user_input)
            
            print("\n" + "="*60)
            print("RESPONSE:")
            print("="*60)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()