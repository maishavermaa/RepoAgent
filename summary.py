import json
import os
import hashlib
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CodeSummary:
    """Represents a summary of related code chunks"""
    summary_id: str
    title: str
    summary_text: str
    file_paths: List[str]
    chunk_ids: List[str]
    keywords: List[str]
    confidence_score: float
    category: str  # e.g., "class", "function", "module", "concept"

class SmartSummaryAgent:
    """
    Agent that creates intelligent summaries from code files and serves as a fast query layer
    """
    
    def __init__(self, openai_api_key: str = None, confidence_threshold: float = 0.6):
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.confidence_threshold = confidence_threshold
        self.query_patterns = defaultdict(int)  # Track common query patterns
        
        # Initialize the code indexer for file-based summaries
        from code_indexer import get_indexed_codebase
        self.indexer = get_indexed_codebase()
        
        # Initialize repo assistant for fallback
        self.repo_assistant = None
        self._init_repo_assistant()
        
        if not self.openai_api_key:
            logger.warning("No OpenAI API key found. AI enhancements will be disabled.")
    
    def _init_repo_assistant(self):
        """Initialize the repository assistant for fallback queries"""
        try:
            from repo_assistant import FileSummaryAssistant
            if self.openai_api_key:
                self.repo_assistant = FileSummaryAssistant(self.openai_api_key)
                if self.repo_assistant.load_file_summaries():
                    logger.info("Repository assistant initialized for fallback queries")
                else:
                    self.repo_assistant = None
            else:
                logger.warning("No OpenAI API key - repo assistant fallback disabled")
        except Exception as e:
            logger.warning(f"Could not initialize repo assistant: {e}")
            self.repo_assistant = None
    
    def regenerate_all_summaries_with_ai(self, use_parallel: bool = True, max_concurrent: int = 15) -> bool:
        """Regenerate all summaries using AI-powered analysis"""
        if not self.indexer:
            logger.error("No code indexer available. Please ingest a repository first.")
            return False
        
        if not self.openai_api_key:
            logger.error("OpenAI API key required for AI-powered summary regeneration")
            return False
        
        processing_mode = "PARALLEL" if use_parallel else "SEQUENTIAL"
        logger.info(f"Starting AI-powered summary regeneration using {processing_mode} processing...")
        
        # Get current count
        count = self.indexer.collection.count()
        if count == 0:
            logger.error("No existing summaries found to regenerate")
            return False
        
        logger.info(f"Found {count} summaries to regenerate with AI")
        
        # Get all existing data
        all_data = self.indexer.collection.get()
        file_paths = [meta['file_path'] for meta in all_data['metadatas']]
        
        # Clear existing summaries
        try:
            self.indexer.collection.delete(ids=all_data['ids'])
            logger.info("Cleared existing summaries")
        except Exception as e:
            logger.error(f"Could not clear existing summaries: {e}")
            return False
        
        # Find the original source directory by looking at file paths
        import os
        if file_paths:
            # Look for data/repositories directory or try to infer from paths
            possible_roots = []
            
            # Check if files are from data/repositories
            for path in file_paths[:5]:  # Check first few paths
                if '/' in path:
                    parts = path.split('/')
                    if len(parts) > 1:
                        # Try data/repositories/repo_name format
                        possible_root = os.path.join('data', 'repositories', parts[0])
                        if os.path.exists(possible_root):
                            possible_roots.append(possible_root)
            
            # Try most common root
            if possible_roots:
                from collections import Counter
                most_common_root = Counter(possible_roots).most_common(1)[0][0]
                logger.info(f"Re-ingesting from detected root: {most_common_root}")
                
                try:
                    # Use parallel or sequential processing based on parameter
                    from code_indexer import CodeIndexer
                    new_indexer = CodeIndexer(max_concurrent=max_concurrent)
                    
                    if use_parallel:
                        logger.info(f"Using parallel processing with {max_concurrent} concurrent requests")
                        indexed_files = new_indexer.ingest_directory_with_parallel_ai(most_common_root)
                    else:
                        logger.info("Using sequential processing")
                        indexed_files = new_indexer.ingest_directory(most_common_root)
                    
                    logger.info(f"Successfully regenerated {len(indexed_files)} AI-powered summaries")
                    
                    # Update our indexer reference
                    self.indexer = new_indexer
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to re-ingest directory {most_common_root}: {e}")
        
        logger.error("Could not find source files to regenerate summaries")
        logger.info("To regenerate summaries, please re-ingest your repository:")
        logger.info("python3 chatbot.py --zip <your-repo.zip> --parallel  # For fast parallel processing")
        return False
    
    def _evaluate_summary_confidence(self, query: str, summaries: List[Dict]) -> float:
        """Evaluate if summaries are detailed enough to answer the query"""
        if not summaries:
            return 0.0
        
        # Calculate confidence based on summary quality and coverage
        total_confidence = 0.0
        query_lower = query.lower()
        
        # Keywords that indicate need for detailed analysis
        detail_keywords = ['how', 'what', 'why', 'explain', 'describe', 'show', 'implement', 'works', 'configure', 'setup']
        code_keywords = ['class', 'function', 'method', 'interface', 'api', 'endpoint', 'authentication', 'database']
        
        for summary in summaries:
            summary_text = summary.get('summary', '').lower()
            
            # Higher confidence for longer, more detailed summaries
            length_score = min(len(summary_text) / 1000, 1.0)  # Up to 1000 chars = full score
            
            # Check if summary contains structured information (our new format)
            structure_score = 0.0
            if any(marker in summary_text for marker in ['**', 'purpose:', 'methods:', 'dependencies:', 'use case:']):
                structure_score = 0.4  # High score for structured summaries
            
            # Check keyword relevance
            relevance_score = 0.0
            query_words = query_lower.split()
            matching_words = sum(1 for word in query_words if word in summary_text)
            if query_words:
                relevance_score = matching_words / len(query_words)
            
            # Check for technical detail indicators
            detail_score = 0.0
            if any(keyword in summary_text for keyword in code_keywords):
                detail_score = 0.3
            
            # Combine scores
            summary_confidence = (length_score * 0.3 + structure_score + relevance_score * 0.2 + detail_score)
            total_confidence += summary_confidence
        
        # Average confidence across all summaries, but boost if we have multiple relevant summaries
        avg_confidence = total_confidence / len(summaries)
        
        # Boost confidence if we have multiple relevant summaries
        if len(summaries) > 1:
            avg_confidence *= 1.2
        
        # Cap at 1.0
        return min(avg_confidence, 1.0)
    
    def query_with_summary_first(self, query: str, max_results: int = 5) -> tuple[str, bool]:
        """
        Query using summaries first, with intelligent fallback to repository search
        
        Returns: (response, used_cache)
        """
        logger.info(f"Querying with summary-first approach: {query}")
        
        # Search summaries
        summary_results = self.indexer.search(query, max_results)
        
        # Evaluate if summaries are sufficient
        confidence = self._evaluate_summary_confidence(query, summary_results)
        
        logger.info(f"Summary confidence score: {confidence:.2f} (threshold: {self.confidence_threshold})")
        
        # Use summaries if confidence is high enough OR if they contain structured data
        structured_indicators = ['**', 'Methods:', 'Dependencies:', 'Use Case:', 'Purpose:']
        has_structured_summaries = any(
            any(indicator in result.get('summary', '') for indicator in structured_indicators)
            for result in summary_results
        )
        
        # Lower threshold for structured summaries since they're much more detailed
        effective_threshold = 0.4 if has_structured_summaries else self.confidence_threshold
        
        if confidence >= effective_threshold or has_structured_summaries:
            # Generate response from summaries
            context = self._build_summary_context(summary_results, query)
            response = self._generate_summary_response(query, context)
            
            # Track successful summary usage
            self.query_patterns[query] += 1
            
            logger.info(f"✅ Answered using summaries (confidence: {confidence:.2f})")
            return response, True
        else:
            logger.info(f"❌ Summary confidence too low ({confidence:.2f}), falling back to repository search")
            return self._fallback_to_repository(query), False
    
    def _build_summary_context(self, summaries: List[Dict], query: str) -> str:
        """Build a context string from multiple summaries"""
        context = ""
        for summary in summaries:
            context += f"**Summary:** {summary['summary']}\n"
            context += f"**File:** {summary['file_path']}\n"
            context += f"**Type:** {summary['file_type']} ({summary['language']})\n"
            context += f"**Lines:** {summary['line_count']} | **Complexity:** {summary['complexity_score']}\n"
            context += f"**Purpose:** {summary.get('purpose', 'No purpose specified')}\n"
            context += f"**Methods:** {summary.get('methods', 'No methods specified')}\n"
            context += f"**Dependencies:** {summary.get('dependencies', 'No dependencies specified')}\n"
            context += f"**Use Case:** {summary.get('use_case', 'No use case specified')}\n"
            context += f"**Purpose:** {summary.get('purpose', 'No purpose specified')}\n"
            context += f"**Keywords:** {', '.join(summary['keywords'])}\n"
            context += f"**Confidence:** {summary['confidence_score']:.2f}\n"
            context += f"**Category:** {summary['category']}\n"
            context += "\n"
        return context
    
    def _generate_summary_response(self, query: str, context: str) -> str:
        """Generate a response using the built summary context"""
        return f"📋 **Summary Response**\n\n{context}"
    
    def _fallback_to_repository(self, query: str) -> str:
        """Fallback to the repository assistant for detailed analysis"""
        if self.repo_assistant:
            try:
                logger.info("Using repository assistant for detailed analysis")
                return self.repo_assistant.query_code(query)
            except Exception as e:
                logger.error(f"Repository assistant failed: {e}")
        
        # Final fallback message
        return (
            f"🔍 **Repository Analysis**\n\n"
            f"I couldn't find a good summary for your query: '{query}'\n\n"
            f"**Suggestions:**\n"
            f"• Try rephrasing your question with specific keywords\n"
            f"• Ask about specific files, functions, or concepts\n"
            f"• Use 'python3 regenerate_ai_summaries.py' to improve summaries\n\n"
            f"**Example queries:**\n"
            f"• 'What does the config.yaml file configure?'\n"
            f"• 'Explain the main classes in this project'\n"
            f"• 'How does authentication work?'"
        )
    
    def get_summary_stats(self) -> Dict:
        """Get statistics about the current summaries"""
        if not self.indexer:
            return {"message": "No summaries available - please ingest a repository first"}
        
        try:
            count = self.indexer.collection.count()
            if count == 0:
                return {"message": "No summaries found"}
            
            # Get sample of data to analyze
            sample_data = self.indexer.collection.get(limit=min(count, 100))
            
            if sample_data and sample_data['metadatas']:
                file_types = defaultdict(int)
                languages = defaultdict(int)
                complexity_scores = []
                
                for meta in sample_data['metadatas']:
                    file_types[meta.get('file_type', 'unknown')] += 1
                    languages[meta.get('language', 'unknown')] += 1
                    complexity_scores.append(meta.get('complexity_score', 0))
                
                avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
                
                return {
                    "total_summaries": count,
                    "file_types": dict(file_types),
                    "languages": dict(languages),
                    "average_complexity": round(avg_complexity, 1),
                    "most_common_queries": dict(self.query_patterns.most_common(5)),
                    "confidence_threshold": self.confidence_threshold,
                    "ai_enabled": bool(self.openai_api_key),
                    "repo_assistant_available": bool(self.repo_assistant)
                }
            
        except Exception as e:
            logger.error(f"Error getting summary stats: {e}")
        
        return {"message": f"Error accessing summary database: {e}"}

    # Legacy compatibility methods (kept for backward compatibility)
    def analyze_chunks_and_create_summaries(self, chunks_file: str = "repo_chunks.json") -> None:
        """Legacy method - now redirects to AI summary regeneration"""
        logger.warning("analyze_chunks_and_create_summaries is deprecated. Use regenerate_all_summaries_with_ai() instead.")
        self.regenerate_all_summaries_with_ai()
    
    def save_summaries(self):
        """Legacy method - file summaries are automatically saved in ChromaDB"""
        logger.info("File summaries are automatically saved in ChromaDB vector database")
    
    def load_summaries(self):
        """Legacy method - file summaries are automatically loaded from ChromaDB"""
        logger.info("File summaries are automatically loaded from ChromaDB vector database")


# Usage example and integration
def integrate_with_existing_system():
    """
    Example of how to integrate this with your existing MCP server
    """
    
    # Initialize the smart summary agent
    agent = SmartSummaryAgent(confidence_threshold=0.8)
    
    # Create summaries from existing chunks (run this once)
    # agent.analyze_chunks_and_create_summaries("repo_chunks.json")
    
    # Example of modified query function for your MCP server
    def enhanced_search_code(query: str, max_results: int = 5) -> str:
        """Enhanced search that tries summaries first"""
        
        # Try summary-based response first
        response, used_cache = agent.query_with_summary_first(query, max_results)
        
        if used_cache:
            return response
        else:
            # Fall back to your existing search logic
            # This would be your current indexer.search() call
            return f"Falling back to detailed search for: {query}"
    
    return agent, enhanced_search_code


if __name__ == "__main__":
    # Example usage
    agent = SmartSummaryAgent()
    
    # Create summaries (run once)
    print("Creating intelligent summaries...")
    agent.analyze_chunks_and_create_summaries()
    
    # Test queries
    test_queries = [
        "What does this repository do?",
        "Explain the UserService class",
        "How does authentication work?",
        "Show me the main functions"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        response, used_cache = agent.query_with_summary_first(query)
        print(f"Used cache: {used_cache}")
        print(f"Response: {response}")
    
    # Show stats
    print("\n--- Summary Statistics ---")
    print(json.dumps(agent.get_summary_stats(), indent=2))