#!/usr/bin/env python3

import os
import ssl
import urllib3
import sys
import argparse
import threading
import time
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Handle SSL certificate issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

# DISABLE CHROMADB TELEMETRY to avoid SSL blocking
os.environ['CHROMA_TELEMETRY'] = 'false'
os.environ['ANONYMIZED_TELEMETRY'] = 'false'

from code_indexer import index_zip_file, index_directory, CodeIndexer
import mcp_server
from agent import create_agent

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Code Repository Analysis Chatbot")
    parser.add_argument("--zip", help="Path to zip file containing the repository to analyze")
    parser.add_argument("--dir", help="Path to directory containing the repository to analyze")
    parser.add_argument("--parallel", action="store_true", 
                       help="Use parallel AI processing for much faster indexing (recommended)")
    parser.add_argument("--concurrent", type=int, default=5,
                       help="Number of concurrent AI requests when using --parallel (default: 5)")
    parser.add_argument("--mcp-host", default="127.0.0.1", help="Host for the MCP server")
    parser.add_argument("--mcp-port", type=int, default=8000, help="Port for the MCP server")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--api-key", help="OpenAI API key (alternatively set OPENAI_API_KEY env variable)")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OpenAI API key is required. Set it using --api-key or OPENAI_API_KEY environment variable.")
        return
    
    # Index the repository
    if args.zip or args.dir:
        if args.parallel:
            print(f"🚀 Using PARALLEL AI processing with {args.concurrent} concurrent requests")
            print("This will be much faster than sequential processing!")
            
            # Use parallel processing
            indexer = CodeIndexer(max_concurrent=args.concurrent)
            
            if args.zip:
                if not os.path.exists(args.zip):
                    print(f"Error: Zip file not found: {args.zip}")
                    return
                print(f"Indexing zip file with parallel AI: {args.zip}...")
                # Extract zip and use parallel processing on the directory
                repo_path, temp_dir = indexer.extract_zip(args.zip)
                try:
                    indexer.ingest_directory_with_parallel_ai(repo_path)
                finally:
                    import shutil
                    shutil.rmtree(temp_dir)
            elif args.dir:
                if not os.path.exists(args.dir):
                    print(f"Error: Directory not found: {args.dir}")
                    return
                print(f"Indexing directory with parallel AI: {args.dir}...")
                indexer.ingest_directory_with_parallel_ai(args.dir)
        else:
            print("🐌 Using SEQUENTIAL processing (slower)")
            print("Tip: Use --parallel flag for much faster processing!")
            
            # Use traditional sequential processing
            if args.zip:
                if not os.path.exists(args.zip):
                    print(f"Error: Zip file not found: {args.zip}")
                    return
                print(f"Indexing zip file: {args.zip}...")
                index_zip_file(args.zip)
            elif args.dir:
                if not os.path.exists(args.dir):
                    print(f"Error: Directory not found: {args.dir}")
                    return
                print(f"Indexing directory: {args.dir}...")
                index_directory(args.dir)
    else:
        print("Error: Please provide either --zip or --dir argument.")
        print("\nUsage examples:")
        print("  python3 chatbot.py --dir ./my-project --parallel")
        print("  python3 chatbot.py --zip project.zip --parallel --concurrent 20")
        print("\nFor 658+ files, parallel processing can be ~15x faster:")
        print("  Sequential: ~2.7 hours")
        print("  Parallel:   ~10 minutes")
        return
    
    # Start MCP server in a separate thread
    mcp_url = f"http://{args.mcp_host}:{args.mcp_port}"
    server_thread = threading.Thread(
        target=mcp_server.start_server,
        args=(args.mcp_host, args.mcp_port),
        daemon=True
    )
    server_thread.start()
    
    # Wait a bit for the server to start
    print(f"Starting MCP server on {mcp_url}...")
    time.sleep(2)
    
    # Create the agent
    print(f"Initializing code analyzer agent with model: {args.model}...")
    agent = create_agent(mcp_url=mcp_url, model=args.model)
    
    # Start the chat interface
    print("\n===== Code Repository Analysis Chatbot =====")
    print("Type your questions about the codebase. Type 'exit' or 'quit' to end the session.")
    print("Example questions:")
    print("- What does this repository do?")
    print("- Explain the function 'process_data'")
    print("- Show me the structure of this project")
    print("- How does the authentication system work?")
    print("========================================\n")
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in ["exit", "quit", "bye"]:
                print("Exiting. Goodbye!")
                break
            
            if not query:
                continue
            
            print("\nAgent:", end=" ")
            response = agent.run(query, stream=True)
            
        except KeyboardInterrupt:
            print("\nExiting due to keyboard interrupt.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 