import os
from typing import Optional, Dict, Any, List

from openai_agents import Agent, HostedMCPTool
from openai import OpenAI
from openai.types.responses.tool import Mcp

class CodebaseAnalyzerAgent:
    """Agent for analyzing codebases"""
    
    def __init__(self, mcp_url: str = "http://localhost:8000", model: str = "gpt-4o"):
        """Initialize the agent with the MCP server URL and model"""
        self.mcp_url = mcp_url
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Define the MCP tool with correct structure
        mcp_config = Mcp(
            server_label="codebase_analyzer", 
            server_url=self.mcp_url,
            type="mcp"
        )
        
        self.mcp_tool = HostedMCPTool(tool_config=mcp_config)
        
        # Add a hook for tool calls
        original_call = self.mcp_tool.call
        def call_with_logging(*args, **kwargs):
            tool_name = kwargs.get('name', 'unknown_tool')
            print(f"AGENT: Calling tool '{tool_name}' with args: {kwargs.get('arguments', {})}")
            return original_call(*args, **kwargs)
        
        self.mcp_tool.call = call_with_logging
        
        # Define the agent
        self.agent = Agent(
            name="CodebaseAnalyzer",
            system_prompt="""You are an expert code assistant that helps users understand and navigate a codebase.
The codebase has been indexed using file-summary based indexing, where each file is analyzed and summarized with:
- File purpose and overview
- Key classes and their methods  
- Key functions and their parameters
- Important imports and dependencies
- File type classification (class-based, function-based, mixed, script, config, documentation)
- Line count and complexity metrics

You have access to tools to explore the code repository:

1. search_code - Search for relevant file summaries based on a query (returns file metadata, not code chunks)
2. explain_function - Explain a specific function, including its purpose, parameters, and return type
3. get_file_content - Get the complete content of a specific file
4. list_project_structure - Show the directory structure of the project
5. explain_class - Explain a class, including its methods, properties, and inheritance

IMPORTANT WORKFLOW:
- Start with list_project_structure to understand the codebase organization
- Use search_code to find relevant files (this returns file summaries, not code snippets)
- Use get_file_content to examine specific files in detail when needed
- Use explain_function/explain_class for detailed analysis of specific code elements

The search results will show file summaries with metadata like:
- File type, language, and complexity
- Number of classes and functions
- File purpose and main components
- Import dependencies

When explaining code, provide comprehensive explanations that highlight the purpose, architecture patterns, 
and how different files work together. Since you work with complete file context rather than fragments,
you can provide more coherent and contextually aware responses.

Your goal is to help users understand the overall codebase architecture and specific implementation details.
""",
            tools=[self.mcp_tool],
            model=self.model,
            tool_choice="auto",
        )
    
    def run(self, query: str, stream: bool = False) -> str:
        """Run the agent with a query"""
        print(f"AGENT: Running with query: '{query}'")
        try:
            if stream:
                response_stream = self.agent.run_stream(query)
                full_response = ""
                for chunk in response_stream:
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                        full_response += chunk.content
                print()  # Add a newline at the end
                return full_response
            else:
                print("AGENT: Starting non-streaming response")
                response = self.agent.run(query)
                print("AGENT: Completed response")
                return response
        except Exception as e:
            print(f"AGENT ERROR: {str(e)}")
            return f"Error running agent: {str(e)}"


def create_agent(mcp_url: str = "http://localhost:8000", model: str = "gpt-4o") -> CodebaseAnalyzerAgent:
    """Create a new instance of the codebase analyzer agent"""
    return CodebaseAnalyzerAgent(mcp_url=mcp_url, model=model) 