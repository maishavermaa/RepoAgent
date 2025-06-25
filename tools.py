# tools.py

import os
import zipfile
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import re

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_id: str
    file_type: str

class RepositoryIngestor:
    """Handles repository ingestion and chunking, and provides search functionality."""
    
    # Files and directories to ignore (common hidden/system files)
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
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        # self.chunks is not used here as chunks are passed directly to search_code_in_chunks
    
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
    
    def extract_zip(self, zip_path: str, extract_to: str) -> str:
        """Extract zip file and return extraction directory"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Find the root directory (usually the repo name)
        extracted_items = os.listdir(extract_to)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(extract_to, extracted_items[0])):
            return os.path.join(extract_to, extracted_items[0])
        return extract_to
    
    def chunk_file_content(self, file_path: str, content: str) -> List[CodeChunk]:
        """Split file content into chunks"""
        lines = content.split('\n')
        chunks = []
        
        for i in range(0, len(lines), self.chunk_size):
            chunk_lines = lines[i:i + self.chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            # Create unique chunk ID
            chunk_id = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()[:8]
            
            chunk = CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=i + 1,
                end_line=min(i + self.chunk_size, len(lines)),
                chunk_id=chunk_id,
                file_type=Path(file_path).suffix
            )
            chunks.append(chunk)
        
        return chunks
    
    def ingest_repository(self, repo_path: str) -> List[CodeChunk]:
        """Ingest repository and return list of chunks"""
        all_chunks: List[CodeChunk] = []
        
        # If it's a zip file, extract it first
        if repo_path.endswith('.zip'):
            # Use a more robust temporary directory for extraction
            temp_extract_base = Path(os.path.dirname(repo_path)) / "extracted_repos"
            temp_extract_base.mkdir(parents=True, exist_ok=True)
            
            # Create a unique directory for this extraction to avoid conflicts
            extract_to = temp_extract_base / Path(repo_path).stem
            extract_to.mkdir(exist_ok=True) # Ensure it exists if it's the same name
            
            repo_path_actual = self.extract_zip(repo_path, str(extract_to))
        else:
            repo_path_actual = repo_path
        
        # Walk through all files
        for root, dirs, files in os.walk(repo_path_actual):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore(os.path.join(root, d))]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path_actual)
                
                # Skip ignored files
                if self.should_ignore(file_path) or not self.is_code_file(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    file_chunks = self.chunk_file_content(relative_path, content)
                    all_chunks.extend(file_chunks)
                    
                except Exception as e:
                    print(f"Warning: Could not read {relative_path}: {e}")
        
        print(f"Ingested {len(all_chunks)} chunks from repository")
        return all_chunks

    def search_code_in_chunks(self, chunks: List[CodeChunk], search_query: str, max_chunks_to_return: int = 15) -> List[CodeChunk]:
        """
        Search for chunks relevant to the query using keyword matching.
        This function is designed to be called by the LLM tool, operating on a given list of chunks.
        """
        query_lower = search_query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        chunk_scores = []
        
        for chunk in chunks: # Operate on the passed chunks list
            content_lower = chunk.content.lower()
            file_path_lower = chunk.file_path.lower()
            
            score = 0
            # Boost score if query words are found
            for word in query_words:
                score += content_lower.count(word) * 2
                score += file_path_lower.count(word)
            
            score += 0.01 # Add a small score just for existing to make sure they are considered

            if score > 0:
                chunk_scores.append((chunk, score))
        
        chunk_scores.sort(key=lambda x: x[1], reverse=True)

        if not chunk_scores and chunks:
            # If no scores and chunks exist, return a broad selection
            return chunks[:max_chunks_to_return] 
        
        return [chunk for chunk, _ in chunk_scores[:max_chunks_to_return]]