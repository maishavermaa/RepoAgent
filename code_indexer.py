import os
import sys
import zipfile
import tempfile
import shutil
import json
import ast
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from pathlib import Path
import hashlib
import asyncio
import aiohttp
import time

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a singleton instance of the code indexer
_CODE_INDEXER_INSTANCE = None

class FileSummary:
    """Represents a comprehensive AI-generated summary of a code file"""
    def __init__(self, file_path: str, content: str, language: str = None, openai_client=None):
        self.file_path = file_path
        self.content = content
        self.language = language or self._detect_language(file_path)
        self.line_count = len(content.split('\n'))
        self.openai_client = openai_client
        
        # Initialize summary components
        self.ai_summary = ""
        self.purpose = ""
        self.key_components = []
        self.file_type = "unknown"
        self.complexity_score = 0
        self.technical_details = {}
        self.ai_generated = False
        
        # Generate the AI-powered summary
        self._generate_ai_summary()
        self._extract_metadata_from_summary()
        
    def _detect_language(self, file_path):
        """Detect the programming language based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.sh': 'bash',
            '.json': 'json',
            '.md': 'markdown',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.xml': 'xml',
            '.txt': 'text',
            '.sql': 'sql',
            '.dockerfile': 'docker',
            '.makefile': 'make',
            '.cmake': 'cmake',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'config',
        }
        return language_map.get(ext, 'text')
    
    def _generate_ai_summary(self):
        """Generate a comprehensive AI-powered summary of the file content"""
        if not self.openai_client:
            # Fallback to basic analysis if no OpenAI client
            self._fallback_basic_analysis()
            return
        
        try:
            # Use larger content window for detailed analysis
            content_preview = self.content[:12000] if len(self.content) > 12000 else self.content
            truncated_notice = "...\n[Content truncated for analysis]" if len(self.content) > 12000 else ""
            
            # Create a comprehensive prompt based on file type
            prompt = self._create_analysis_prompt(content_preview + truncated_notice)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use the better model for detailed summaries
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert code analyst. Provide extremely detailed, technical summaries of code files. Use structured formatting with clear sections. Include all method signatures, class details, dependencies, and use cases. Be comprehensive and thorough."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1500,  # Much larger token limit for detailed summaries
                temperature=0.1
            )
            
            self.ai_summary = response.choices[0].message.content.strip()
            logger.info(f"Generated detailed AI summary for {self.file_path}")
            
        except Exception as e:
            logger.warning(f"AI summary generation failed for {self.file_path}: {e}")
            self._fallback_basic_analysis()
    
    def _create_analysis_prompt(self, content: str) -> str:
        """Create a tailored analysis prompt based on file type and language"""
        
        base_prompt = f"""Analyze this {self.language} file ({self.line_count} lines) and provide an extremely comprehensive, detailed textual summary.

{content}

"""

        if self.language == 'java':
            return base_prompt + """Create a comprehensive textual summary following this EXACT format:

Summary of [ClassName/InterfaceName]
Package: [package.name]

Purpose: [Detailed description of what this file does and its role in the system]

Class/Interface: [Name] ([Type - class/interface/enum/abstract class])
[If it extends or implements anything, mention it]

Methods:
[List EVERY method with full signature and description]
[ReturnType] [methodName]([parameter types and names])
    [Detailed description of what this method does, parameters, return value]

[Continue for ALL methods - do not skip any]

Fields/Properties:
[List ALL fields with types and descriptions]
[type] [fieldName] - [description]

Dependencies:
[List ALL imports and explain what each is used for]
- [import] - [purpose]

Annotations:
[List any annotations used and their purpose]

Use Case:
[Explain where and how this class/interface is used in the system]

Key Features:
[Notable patterns, design principles, algorithms, or special functionality]

Be EXTREMELY thorough. List every single method, field, import, and annotation. Do not skip anything."""

        elif self.language == 'python':
            return base_prompt + """Create a comprehensive textual summary following this EXACT format:

Summary of [FileName]
Module: [module.path]

Purpose: [Detailed description of what this module does and its role]

Classes:
[For EVERY class in the file]
class [ClassName]([BaseClass if any]):
    Purpose: [What this class does]
    
    Methods:
    [List ALL methods with signatures]
    def [method_name](self, [parameters]) -> [return_type]:
        [Detailed description of what this method does]
    
    [Continue for ALL methods in the class]
    
    Properties/Attributes:
    [List ALL class attributes]
    [attribute_name]: [type] - [description]

Functions:
[For EVERY standalone function]
def [function_name]([parameters]) -> [return_type]:
    [Detailed description of what this function does]

Variables/Constants:
[List ALL module-level variables and constants]
[variable_name] = [value] - [description]

Dependencies:
[List ALL imports and their purposes]
import [module] - [what it's used for]
from [module] import [items] - [what they're used for]

Use Case:
[How this module fits in the larger system]

Key Features:
[Design patterns, algorithms, notable logic, decorators used]

Be EXTREMELY comprehensive. List every function, class, method, import, and variable."""

        elif self.language in ['javascript', 'typescript']:
            return base_prompt + """Create a comprehensive textual summary following this EXACT format:

Summary of [FileName]
Module: [module/component name]

Purpose: [What this file does and its role]

Exports:
[List EVERYTHING this file exports]
- [export_name]: [type] - [description]

Functions:
[For EVERY function]
function [functionName]([parameters]): [returnType]
    [Detailed description including parameters and return value]

Classes:
[For EVERY class]
class [ClassName] [extends/implements]:
    Purpose: [What this class does]
    
    Methods:
    [method_name]([parameters]): [returnType]
        [Description of what this method does]

Components: (if React/Vue)
[For EVERY component]
[ComponentName]:
    Purpose: [What this component does]
    Props: [list all props with types]
    State: [state management details]

Variables/Constants:
[List ALL module-level variables and constants]
const/let/var [name] = [value] - [description]

Dependencies:
[List ALL imports and their purposes]
import [items] from '[module]' - [what they're used for]

APIs/Endpoints:
[Any API calls, fetch requests, or endpoints defined]

Event Handlers:
[Any event listeners or handlers]

Use Case:
[Role in the application]

Key Features:
[Patterns, algorithms, frameworks used]

List every function, class, variable, import, and export. Be extremely thorough."""

        elif self.language in ['yaml', 'json', 'toml', 'ini', 'config']:
            return base_prompt + """Create a comprehensive textual summary following this EXACT format:

Configuration File Summary
File: [filename]
Type: [Configuration type - application config, database config, deployment, etc.]

Purpose: [What system/application this configures and why]

Configuration Structure:
[Analyze the ENTIRE configuration file structure]

Main Sections:
[For EVERY major section]
[section_name]:
    Purpose: [What this section configures]
    Settings:
    [List EVERY setting in this section]
    - [setting_name]: [value] - [detailed explanation of what this does]

Environment Configuration:
[Any environment-specific settings]

Service Configuration:
[Any services being configured]

Database Configuration:
[Database connections, settings]

Security Configuration:
[Authentication, authorization, encryption settings]

Network Configuration:
[Ports, hosts, URLs, endpoints]

Performance Configuration:
[Caching, timeouts, memory settings, limits]

Logging Configuration:
[Log levels, outputs, formats]

Feature Flags:
[Any feature toggles or flags]

Dependencies:
[External systems, services, or databases configured]

Environment Variables:
[Any environment variables referenced]

Use Case:
[When and how this configuration is used in deployment/runtime]

Impact:
[How changing these settings affects the system]

Be extremely detailed. Explain every single configuration option and its impact."""

        elif self.language == 'markdown':
            return base_prompt + """Create a comprehensive textual summary following this EXACT format:

Documentation Summary
Document: [title/main topic]

Purpose: [What this documentation covers and who it's for]

Structure:
[Analyze the complete document structure]

Sections:
[For EVERY section and subsection]
# [Section Title]
    Content: [what is covered in detail]
    Key Points:
    - [important point 1]
    - [important point 2]

Instructions/Tutorials:
[List ALL step-by-step guides]
[Tutorial Name]:
    Steps: [summarize the steps]
    Requirements: [prerequisites]
    Outcome: [what you accomplish]

Code Examples:
[List ALL code examples shown]
[Language]: [what the code demonstrates]

API References:
[Any APIs, endpoints, or technical references]

Commands/Scripts:
[Any terminal commands or scripts mentioned]

Links and References:
[ALL external links and what they point to]

Technical Concepts:
[Key concepts, terms, or technologies explained]

Target Audience:
[Who should read this and when]

Use Case:
[When to reference this documentation]

Completeness:
[Assessment of what's covered vs what might be missing]

Be extremely thorough. Capture every section, example, link, and concept."""

        elif self.language in ['sql']:
            return base_prompt + """Create a comprehensive textual summary following this EXACT format:

Database Script Summary
Script: [filename]
Type: [DDL, DML, migration, stored procedures, etc.]

Purpose: [What this script accomplishes in the database]

Tables:
[For EVERY table created/modified]
Table: [table_name]
    Purpose: [what this table stores]
    Columns:
    [List EVERY column]
    - [column_name]: [data_type] [constraints] - [description]
    
    Primary Key: [primary key columns]
    Foreign Keys:
    - [foreign_key] REFERENCES [referenced_table]([referenced_column])
    
    Indexes:
    [List ALL indexes]
    - [index_name] ON [columns] - [purpose]

Views:
[For EVERY view]
View: [view_name]
    Purpose: [what data this view provides]
    Columns: [columns in the view]
    Source Tables: [tables used]

Stored Procedures:
[For EVERY procedure]
PROCEDURE [procedure_name]([parameters])
    Purpose: [what this procedure does]
    Parameters: [parameter descriptions]
    Returns: [what it returns]
    Logic: [key operations performed]

Functions:
[For EVERY function]
FUNCTION [function_name]([parameters]) RETURNS [type]
    Purpose: [what this function does]

Triggers:
[For EVERY trigger]
TRIGGER [trigger_name] ON [table] [timing] [event]
    Purpose: [what this trigger does]

Data Operations:
[ALL INSERT, UPDATE, DELETE operations]
- [operation]: [what data is affected]

Constraints:
[ALL constraints defined]
- [constraint_name]: [type] - [what it enforces]

Dependencies:
[Other database objects referenced]

Use Case:
[When this script is executed and why]

Data Flow:
[How data moves through the operations]

Be extremely detailed about every table, column, procedure, and operation."""

        else:
            return base_prompt + """Create a comprehensive textual summary following this EXACT format:

File Summary
File: [filename]
Language: [programming language]
Type: [file type/purpose]

Purpose: [Detailed description of what this file does]

Structure Analysis:
[Complete breakdown of file structure]

Main Components:
[List EVERY major component]

Functions:
[For EVERY function]
[function_name]([parameters]): [return_type]
    Purpose: [what this function does]
    Parameters: [parameter descriptions]
    Returns: [return value description]

Classes/Structures:
[For EVERY class or structure]
[ClassName]:
    Purpose: [what this class does]
    Methods: [list all methods]
    Properties: [list all properties]

Variables/Constants:
[For EVERY variable and constant]
[variable_name]: [type] - [description and usage]

Dependencies:
[ALL external libraries, modules, or files used]
- [dependency]: [how it's used]

Configuration:
[Any configuration values, settings, or constants]

Algorithms:
[Any significant algorithms or logic]

Data Structures:
[Any important data structures used]

Integration Points:
[How this file connects with other parts]

Input/Output:
[What this file reads from and writes to]

Error Handling:
[How errors are handled]

Use Case:
[Role in the larger application]

Complexity Assessment:
[Assessment of the code complexity and maintainability]

Be extremely comprehensive. List every function, class, variable, and dependency with detailed descriptions."""

    def _fallback_basic_analysis(self):
        """Fallback to basic analysis when AI is not available"""
        self.ai_summary = f"Basic analysis: {self.language.title()} file with {self.line_count} lines. "
        
        if self.language in ['json', 'yaml', 'toml']:
            self.ai_summary += "Configuration file - requires manual review for detailed analysis."
        elif self.language == 'markdown':
            # Extract first few meaningful lines for markdown
            lines = [line.strip() for line in self.content.split('\n')[:10] if line.strip()]
            if lines:
                self.ai_summary += f"Documentation starting with: {' '.join(lines[:3])[:100]}..."
        elif self.language in ['python', 'javascript', 'java']:
            self.ai_summary += "Code file - basic structure analysis available but AI summary recommended."
        else:
            self.ai_summary += "Text-based file - content analysis requires manual review."
    
    def _extract_metadata_from_summary(self):
        """Extract metadata from the AI summary for classification"""
        summary_lower = self.ai_summary.lower()
        
        # Determine file type based on content and summary
        if self.language in ['json', 'yaml', 'xml', 'toml', 'ini', 'cfg']:
            self.file_type = "configuration"
        elif self.language == 'markdown':
            self.file_type = "documentation"
        elif 'class' in summary_lower and 'function' in summary_lower:
            self.file_type = "mixed_code"
        elif 'class' in summary_lower:
            self.file_type = "class_based"
        elif 'function' in summary_lower:
            self.file_type = "functional"
        elif self.language == 'sql':
            self.file_type = "database"
        elif self.language == 'docker':
            self.file_type = "container"
        else:
            self.file_type = "script"
        
        # Extract purpose (first sentence or up to first period)
        sentences = self.ai_summary.split('.')
        if sentences:
            self.purpose = sentences[0].strip() + '.'
        else:
            self.purpose = self.ai_summary[:100] + "..." if len(self.ai_summary) > 100 else self.ai_summary
        
        # Calculate complexity based on content and summary analysis
        self._calculate_ai_complexity()
    
    def _calculate_ai_complexity(self):
        """Calculate complexity score based on AI analysis and content"""
        summary_lower = self.ai_summary.lower()
        
        # Base complexity from file size
        base_score = min(self.line_count // 25, 15)
        
        # Add complexity based on summary content
        complexity_indicators = [
            ('multiple classes', 5),
            ('inheritance', 3),
            ('design pattern', 4),
            ('algorithm', 3),
            ('database', 2),
            ('api', 2),
            ('authentication', 3),
            ('configuration', 1),
            ('complex logic', 4),
            ('state management', 3),
            ('async', 2),
            ('threading', 4),
            ('security', 2),
            ('performance', 2),
        ]
        
        for indicator, score in complexity_indicators:
            if indicator in summary_lower:
                base_score += score
        
        self.complexity_score = min(base_score, 25)  # Cap at 25
    
    def to_summary_text(self) -> str:
        """Generate comprehensive text summary for vector embedding (without redundant file path)"""
        # Just return the AI summary - file path is stored separately in metadata
        return self.ai_summary
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        return {
            'file_path': self.file_path,
            'language': self.language,
            'line_count': self.line_count,
            'purpose': self.purpose,
            'ai_summary': self.ai_summary,
            'file_type': self.file_type,
            'complexity_score': self.complexity_score,
            'summary_text': self.to_summary_text()
        }

    @classmethod
    def create_basic_summary(cls, file_path: str, content: str) -> 'FileSummary':
        """Create a basic FileSummary without AI processing"""
        summary = cls.__new__(cls)
        summary.file_path = file_path
        summary.content = content
        summary.language = summary._detect_language(file_path)
        summary.line_count = len(content.split('\n'))
        summary.openai_client = None
        
        # Initialize summary components
        summary.ai_summary = ""
        summary.purpose = ""
        summary.key_components = []
        summary.file_type = "unknown"
        summary.complexity_score = 0
        summary.technical_details = {}
        summary.ai_generated = False
        
        # Use fallback analysis instead of AI
        summary._fallback_basic_analysis()
        summary._extract_metadata_from_summary()
        
        return summary

class CodeIndexer:
    """Handles code indexing and searching using vector embeddings of file summaries"""
    
    # Files and directories to ignore
    IGNORE_PATTERNS = {
        '.DS_Store', '.env', '.git', '.gitignore', '.vscode', '.idea',
        '__pycache__', '.pytest_cache', 'node_modules', '.npm',
        '.Trash', '.Spotlight-V100', '.fseventsd', '.DocumentRevisions-V100',
        '.TemporaryItems', '.apdisk', 'Thumbs.db', 'Desktop.ini'
    }
    
    # File extensions to process
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
        '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
        '.sh', '.bash', '.zsh', '.ps1', '.sql', '.html', '.css', '.scss',
        '.less', '.xml', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
        '.md', '.rst', '.txt', '.dockerfile', '.makefile', '.cmake'
    }
    
    def __init__(self, db_directory="./vector_db", openai_api_key=None, max_concurrent=15):
        self.db_directory = db_directory
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.collection_name = "file_summaries"  # Store collection name for access
        self.max_concurrent = max_concurrent
        
        # Initialize OpenAI client for AI summaries
        self.openai_client = None
        if self.openai_api_key:
            try:
                from openai import OpenAI
                
                # Create HTTP client that ignores SSL verification for debugging
                http_client = httpx.Client(verify=False)
                
                self.openai_client = OpenAI(
                    api_key=self.openai_api_key,
                    http_client=http_client
                )
                logger.info("OpenAI client initialized for AI-powered summaries (SSL verification disabled for debugging)")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}. Will use basic summaries.")
        
        # Initialize the vector database client
        self.client = chromadb.PersistentClient(path=db_directory)
        
        # Use the OpenAI embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name="text-embedding-3-small"
        )
        
        # Check for and create the collection if it doesn't exist
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection '{self.collection_name}' with {self.collection.count()} documents")
        except (ValueError, Exception):
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=openai_ef
            )
            logger.info(f"Created new collection '{self.collection_name}'")
        
        # Store file contents for direct access
        self.file_contents = {}
        self.all_files = []
    
    def should_ignore(self, path: str) -> bool:
        """Check if a file/directory should be ignored"""
        path_obj = Path(path)
        
        # Check if any part of the path matches ignore patterns
        for part in path_obj.parts:
            if part.startswith('.') and part in self.IGNORE_PATTERNS:
                return True
            if part in self.IGNORE_PATTERNS:
                return True
        
        return False
    
    def is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file we should process"""
        return Path(file_path).suffix.lower() in self.CODE_EXTENSIONS
    
    def extract_zip(self, zip_path: str) -> str:
        """Extract zip file to a temporary directory and return the path"""
        extract_dir = tempfile.mkdtemp()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the root directory (usually the repo name)
        extracted_items = os.listdir(extract_dir)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_dir, extracted_items[0])):
            return os.path.join(extract_dir, extracted_items[0]), extract_dir
        
        return extract_dir, extract_dir
    
    def create_file_summary(self, file_path: str, content: str) -> FileSummary:
        """Create a comprehensive AI-powered summary of a single file"""
        return FileSummary(file_path=file_path, content=content, openai_client=self.openai_client)
    
    def ingest_zip(self, zip_path: str) -> List[str]:
        """Ingest a zip file containing a code repository"""
        # Extract the zip file
        repo_path, temp_dir = self.extract_zip(zip_path)
        
        try:
            # Process the extracted files
            return self.ingest_directory(repo_path)
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
    
    def ingest_directory(self, directory_path: str) -> List[str]:
        """Ingest a directory containing code files"""
        all_summaries = []
        indexed_files = []
        
        # Clear previous data - Use delete with proper where clause or recreate collection
        try:
            # Try to delete all documents
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
        except Exception:
            # If delete fails, recreate the collection
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass
            
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name="text-embedding-3-small"
            )
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=openai_ef
            )
        
        self.file_contents = {}
        self.all_files = []
        
        # Walk through all files
        for root, dirs, files in os.walk(directory_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore(os.path.join(root, d))]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory_path)
                
                # Skip ignored files
                if self.should_ignore(file_path) or not self.is_code_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Store full file content
                    self.file_contents[relative_path] = content
                    self.all_files.append(relative_path)
                    
                    # Create summary for the file
                    file_summary = self.create_file_summary(relative_path, content)
                    all_summaries.append(file_summary)
                    indexed_files.append(relative_path)
                    
                except Exception as e:
                    logger.warning(f"Could not read {relative_path}: {e}")
        
        # Prepare data for the vector database
        ids = []
        texts = []
        metadatas = []
        
        for i, summary in enumerate(all_summaries):
            summary_id = f"file_{i}"
            ids.append(summary_id)
            texts.append(summary.to_summary_text())
            metadatas.append({
                "file_path": summary.file_path,
                "language": summary.language,
                "file_type": summary.file_type,
                "line_count": summary.line_count,
                "complexity_score": summary.complexity_score,
                "purpose": summary.purpose
            })
        
        # Add data to the collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        logger.info(f"Indexed {len(all_summaries)} file summaries from {len(indexed_files)} files")
        return indexed_files
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for file summaries relevant to the query"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        formatted_results = []
        if results and results['documents'] and results['metadatas']:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                formatted_results.append({
                    'file_path': metadata['file_path'],
                    'summary': doc,
                    'language': metadata['language'],
                    'file_type': metadata['file_type'],
                    'line_count': metadata['line_count'],
                    'complexity_score': metadata['complexity_score'],
                    'purpose': metadata['purpose']
                })
        
        return formatted_results
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get the full content of a file"""
        return self.file_contents.get(file_path)
    
    def get_all_files(self) -> List[str]:
        """Get a list of all indexed files"""
        return self.all_files

    def index_file(self, file_path: str, force_reindex: bool = False) -> bool:
        """Index or re-index a single file"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return False
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Create relative path
            relative_path = os.path.relpath(file_path) if os.path.isabs(file_path) else file_path
            
            # Skip if not a code file
            if not self.is_code_file(file_path):
                logger.info(f"Skipping non-code file: {relative_path}")
                return False
            
            # Check if already exists and remove if force_reindex
            if force_reindex:
                try:
                    # Find existing documents with this file path
                    existing = self.collection.get(where={"file_path": relative_path})
                    if existing['ids']:
                        self.collection.delete(ids=existing['ids'])
                        logger.info(f"Removed existing summary for: {relative_path}")
                except Exception as e:
                    logger.warning(f"Could not remove existing summary: {e}")
            
            # Create new summary
            file_summary = self.create_file_summary(relative_path, content)
            
            # Generate unique ID
            summary_id = f"file_{hashlib.md5(relative_path.encode()).hexdigest()}"
            
            # Add to collection
            self.collection.add(
                ids=[summary_id],
                documents=[file_summary.to_summary_text()],
                metadatas=[{
                    "file_path": file_summary.file_path,
                    "language": file_summary.language,
                    "file_type": file_summary.file_type,
                    "line_count": file_summary.line_count,
                    "complexity_score": file_summary.complexity_score,
                    "purpose": file_summary.purpose
                }]
            )
            
            # Update file contents cache
            self.file_contents[relative_path] = content
            if relative_path not in self.all_files:
                self.all_files.append(relative_path)
            
            logger.info(f"Successfully indexed: {relative_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            return False

    async def generate_single_file_summary(self, file_path: str, content: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Tuple[str, str]:
        """Generate AI summary for a single file"""
        async with semaphore:  # Limit concurrent requests
            try:
                # Truncate very large files to save tokens
                if len(content) > 6000:
                    content = content[:6000] + "\n... [truncated]"
                
                # Detect language for the file
                language = self._detect_language_for_file(file_path)
                
                prompt = f"""Analyze this {language} code file and provide a comprehensive summary.

File: {file_path}

Code:
{content}

Provide a detailed summary covering:
1. Main purpose and functionality
2. Key classes, functions, or components
3. Dependencies and imports
4. Code complexity and structure
5. Role in the overall system

Be specific and technical. Focus on what this file actually does."""

                payload = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.1
                }
                
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    ssl=False
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        summary = data['choices'][0]['message']['content']
                        logger.info(f"✓ Generated summary for {file_path}")
                        return file_path, summary
                    else:
                        error_text = await response.text()
                        logger.warning(f"✗ API error for {file_path}: {response.status} - {error_text}")
                        return file_path, f"Error: Failed to generate summary (HTTP {response.status})"
                        
            except asyncio.TimeoutError:
                logger.warning(f"✗ Timeout for {file_path}")
                return file_path, "Error: Request timed out"
            except Exception as e:
                logger.warning(f"✗ Exception for {file_path}: {str(e)}")
                return file_path, f"Error: {str(e)}"

    async def generate_all_summaries_parallel(self, file_data: List[Tuple[str, str]]) -> Dict[str, str]:
        """Generate summaries for all files in parallel"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create SSL-disabled session
        connector = aiohttp.TCPConnector(ssl=False, limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout per request
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            
            # Create tasks for all files
            tasks = [
                self.generate_single_file_summary(file_path, content, session, semaphore)
                for file_path, content in file_data
            ]
            
            logger.info(f"Starting parallel processing of {len(tasks)} files with max {self.max_concurrent} concurrent requests...")
            start_time = time.time()
            
            # Process all files concurrently with progress tracking
            results = {}
            completed = 0
            
            # Process in chunks to show progress
            chunk_size = 50
            for i in range(0, len(tasks), chunk_size):
                chunk_tasks = tasks[i:i + chunk_size]
                chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                
                for result in chunk_results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed: {result}")
                    else:
                        file_path, summary = result
                        results[file_path] = summary
                        completed += 1
                
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = len(file_data) - completed
                eta = remaining / rate if rate > 0 else 0
                
                logger.info(f"Progress: {completed}/{len(file_data)} files ({completed/len(file_data)*100:.1f}%) - "
                          f"Rate: {rate:.1f}/sec - ETA: {eta/60:.1f} minutes")
            
            total_time = time.time() - start_time
            logger.info(f"Completed all {len(results)} summaries in {total_time/60:.1f} minutes "
                       f"(avg {total_time/len(results):.2f}s per file)")
            
            return results

    def _detect_language_for_file(self, file_path: str) -> str:
        """Detect programming language based on file extension"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.css': 'css',
            '.html': 'html',
            '.xml': 'xml',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text',
            '.sql': 'sql',
            '.sh': 'bash',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
        }
        return language_map.get(ext, 'text')

    def ingest_directory_with_parallel_ai(self, directory_path: str) -> List[str]:
        """Ingest directory with parallel AI summary generation"""
        
        # Step 1: Collect all files (fast)
        logger.info("Collecting files...")
        file_data = []
        indexed_files = []
        
        # Clear previous data
        try:
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
        except Exception:
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass
            
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name="text-embedding-3-small"
            )
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=openai_ef
            )
        
        self.file_contents = {}
        self.all_files = []
        
        for root, dirs, files in os.walk(directory_path):
            dirs[:] = [d for d in dirs if not self.should_ignore(os.path.join(root, d))]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory_path)
                
                if (self.should_ignore(file_path) or 
                    not self.is_code_file(file_path) or 
                    '__MACOSX' in relative_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if content.strip():
                        self.file_contents[relative_path] = content
                        self.all_files.append(relative_path)
                        file_data.append((relative_path, content))
                        indexed_files.append(relative_path)
                        
                except Exception as e:
                    logger.warning(f"Could not read {relative_path}: {e}")
        
        logger.info(f"Found {len(file_data)} files to process with AI")
        
        # Step 2: Generate AI summaries in parallel
        if file_data and self.openai_api_key:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ai_summaries = loop.run_until_complete(
                    self.generate_all_summaries_parallel(file_data)
                )
            finally:
                loop.close()
        else:
            logger.warning("No OpenAI API key available - using basic summaries")
            ai_summaries = {}
        
        # Step 3: Create FileSummary objects with AI summaries
        logger.info("Creating file summaries with AI insights...")
        all_summaries = []
        
        for file_path, content in file_data:
            ai_summary = ai_summaries.get(file_path, "No AI summary available")
            
            # Create enhanced FileSummary with AI insights
            file_summary = self.create_file_summary_with_ai_insights(
                file_path, content, ai_summary
            )
            all_summaries.append(file_summary)
        
        # Step 4: Add to vector database
        logger.info("Adding summaries to vector database...")
        self.add_summaries_to_vector_db(all_summaries)
        
        logger.info(f"Successfully indexed {len(all_summaries)} files with individual AI summaries")
        return indexed_files
    
    def create_file_summary_with_ai_insights(self, file_path: str, content: str, ai_summary: str) -> 'FileSummary':
        """Create FileSummary enhanced with AI insights"""
        # Start with basic analysis
        basic_summary = FileSummary.create_basic_summary(file_path, content)
        
        # Enhance with AI summary
        basic_summary.ai_summary = ai_summary
        basic_summary.ai_generated = True
        
        # Try to extract structured info from AI summary
        if "purpose:" in ai_summary.lower() or "functionality:" in ai_summary.lower():
            lines = ai_summary.split('\n')
            for line in lines:
                if 'purpose' in line.lower() and ':' in line:
                    basic_summary.purpose = line.split(':', 1)[1].strip()
                    break
        
        return basic_summary

    def add_summaries_to_vector_db(self, all_summaries: List['FileSummary']):
        """Add file summaries to the vector database in batches"""
        # Prepare data for the vector database
        ids = []
        texts = []
        metadatas = []
        
        for i, summary in enumerate(all_summaries):
            summary_id = f"file_{i}"
            ids.append(summary_id)
            texts.append(summary.to_summary_text())
            metadatas.append({
                "file_path": summary.file_path,
                "language": summary.language,
                "file_type": summary.file_type,
                "line_count": summary.line_count,
                "complexity_score": summary.complexity_score,
                "purpose": summary.purpose
            })
        
        # Add data to the collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

def get_indexed_codebase() -> Optional[CodeIndexer]:
    """Get the singleton instance of the code indexer, or create one if data exists"""
    global _CODE_INDEXER_INSTANCE
    
    if _CODE_INDEXER_INSTANCE:
        return _CODE_INDEXER_INSTANCE
    
    # Try to connect to existing data if no instance exists
    try:
        # Set the environment variable for ChromaDB if it's not set but OPENAI_API_KEY is
        if not os.environ.get('CHROMA_OPENAI_API_KEY') and os.environ.get('OPENAI_API_KEY'):
            os.environ['CHROMA_OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
        
        # Try to create a new indexer and check if data exists
        indexer = CodeIndexer()
        count = indexer.collection.count()
        
        if count > 0:
            _CODE_INDEXER_INSTANCE = indexer
            return _CODE_INDEXER_INSTANCE
        else:
            return None
    except Exception as e:
        logger.error(f"Failed to connect to existing codebase: {e}")
        return None

def index_zip_file(zip_path: str) -> CodeIndexer:
    """Index a zip file and return the code indexer instance"""
    global _CODE_INDEXER_INSTANCE
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    if not _CODE_INDEXER_INSTANCE:
        _CODE_INDEXER_INSTANCE = CodeIndexer()
    
    _CODE_INDEXER_INSTANCE.ingest_zip(zip_path)
    return _CODE_INDEXER_INSTANCE

def index_directory(directory_path: str) -> CodeIndexer:
    """Index a directory and return the code indexer instance"""
    global _CODE_INDEXER_INSTANCE
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not _CODE_INDEXER_INSTANCE:
        _CODE_INDEXER_INSTANCE = CodeIndexer()
    
    _CODE_INDEXER_INSTANCE.ingest_directory(directory_path)
    return _CODE_INDEXER_INSTANCE 