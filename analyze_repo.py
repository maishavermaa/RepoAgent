#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set. Please set it before running this script.")
    sys.exit(1)

try:
    # Import the existing repo_assistant module
    from repo_assistant import main as repo_assistant_main
    
    # Start the repo assistant
    print("Starting Repository Code Assistant...")
    repo_assistant_main()
except Exception as e:
    print(f"Error running the repository assistant: {e}")
    sys.exit(1) 