#!/usr/bin/env python3
"""
Simple viewer for file summaries stored in ChromaDB
"""

import os
import ssl
import urllib3
import fnmatch
from pathlib import Path
from difflib import get_close_matches
from dotenv import load_dotenv

# Handle SSL and disable telemetry
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CHROMA_TELEMETRY'] = 'false'
os.environ['ANONYMIZED_TELEMETRY'] = 'false'

load_dotenv()

from code_indexer import get_indexed_codebase

def main():
    print("🔍 File Summary Viewer")
    print("=" * 50)
    
    indexer = get_indexed_codebase()
    if not indexer:
        print("❌ No summaries found. Run chatbot.py to ingest a repository first.")
        return
    
    total_count = indexer.collection.count()
    print(f"📊 Found {total_count} file summaries")
    
    # Get all data
    all_data = indexer.collection.get()
    documents = all_data['documents']
    metadatas = all_data['metadatas']
    
    # Analyze file types and languages
    file_types = {}
    languages = {}
    complexity_scores = []
    
    for meta in metadatas:
        ft = meta['file_type']
        lang = meta['language']
        file_types[ft] = file_types.get(ft, 0) + 1
        languages[lang] = languages.get(lang, 0) + 1
        complexity_scores.append(meta['complexity_score'])
    
    print(f"\n📁 File Types:")
    for ft, count in sorted(file_types.items()):
        print(f"   {ft}: {count} files")
    
    print(f"\n🔤 Languages:")
    for lang, count in sorted(languages.items()):
        print(f"   {lang}: {count} files")
    
    avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
    print(f"\n📈 Average Complexity Score: {avg_complexity:.1f}")
    
    print(f"\n" + "=" * 50)
    print("🎯 Smart Commands:")
    print("• 'show filename.py' - Show summary for specific file")
    print("• 'show partial_name' - Find files with fuzzy matching")  
    print("• 'ls directory/' - List files in a directory")
    print("• 'find *.java' - Find files by pattern")
    print("• 'search <query>' - Search file content")
    print("• 'recent' - Show recently modified files")
    print("• 'list' - Show all files")
    print("• 'regenerate' - Regenerate summaries")
    print("• 'exit' - Quit")
    print("=" * 50)
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if command.lower() == 'list':
                print(f"\n📋 All {len(metadatas)} files:")
                for i, meta in enumerate(metadatas, 1):
                    print(f"{i:3d}. {meta['file_path']} ({meta['file_type']}, {meta['language']})")
            
            elif command.lower().startswith('search '):
                query = command[7:].strip()
                if query:
                    print(f"\n🔍 Searching for: '{query}'")
                    results = indexer.search(query, 10)
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {result['file_path']}")
                        print(f"   Type: {result['file_type']} | Language: {result['language']}")
                        print(f"   Purpose: {result['purpose']}")
                        print(f"   Complexity: {result['complexity_score']} | Lines: {result['line_count']}")
                        print(f"   Summary: {result['summary'][:200]}...")
                else:
                    print("Usage: search <query>")
            
            elif command.lower().startswith('show '):
                file_pattern = command[5:].strip()
                show_file_summary(metadatas, documents, file_pattern)
            
            elif command.lower().startswith('ls ') or command.lower().startswith('dir '):
                dir_path = command.split(' ', 1)[1].strip() if ' ' in command else ''
                list_directory_files(metadatas, dir_path)
            
            elif command.lower().startswith('find '):
                pattern = command[5:].strip()
                find_files_by_pattern(metadatas, documents, pattern)
            
            elif command.lower() == 'recent':
                show_recent_files(metadatas, documents)
            
            elif command.lower().startswith('regenerate'):
                parts = command.split()
                use_parallel = '--parallel' in parts
                concurrent = 15
                
                # Check for concurrent option
                for i, part in enumerate(parts):
                    if part == '--concurrent' and i + 1 < len(parts):
                        try:
                            concurrent = int(parts[i + 1])
                        except ValueError:
                            print("Invalid concurrent value, using default 15")
                
                if len(parts) == 1 or (len(parts) == 2 and parts[1] == '--parallel'):
                    # Regenerate all summaries
                    mode = "PARALLEL" if use_parallel else "SEQUENTIAL"
                    print(f"🔄 Regenerating ALL summaries with AI-powered analysis using {mode} processing...")
                    if use_parallel:
                        print(f"   Using {concurrent} concurrent requests for faster processing")
                    print("⚠️  This will take several minutes and may use significant OpenAI tokens.")
                    confirm = input("Continue? (y/N): ").lower()
                    if confirm == 'y':
                        count = regenerate_all_summaries(indexer, use_parallel, concurrent)
                        print(f"✅ Regenerated {count} summaries")
                elif len(parts) >= 2:
                    # Regenerate specific file (skip --parallel and --concurrent flags)
                    file_pattern = None
                    for part in parts[1:]:
                        if not part.startswith('--') and not part.isdigit():
                            file_pattern = part
                            break
                    
                    if file_pattern:
                        mode = "PARALLEL" if use_parallel else "SEQUENTIAL"
                        print(f"🔄 Regenerating summaries for files matching: {file_pattern} using {mode} processing")
                        count = regenerate_matching_summaries(indexer, file_pattern, use_parallel, concurrent)
                        print(f"✅ Regenerated {count} summaries")
                    else:
                        print("Usage: regenerate [file_pattern] [--parallel] [--concurrent N]")
                        print("Examples:")
                        print("  regenerate --parallel                    # Regenerate ALL summaries with parallel processing")
                        print("  regenerate *.java --parallel             # Regenerate Java files with parallel processing")
                        print("  regenerate config.yaml                   # Regenerate specific file (sequential)")
                        print("  regenerate --parallel --concurrent 20    # Use 20 concurrent requests")
                else:
                    print("Usage: regenerate [file_pattern] [--parallel] [--concurrent N]")
                    print("Examples:")
                    print("  regenerate --parallel                    # Regenerate ALL summaries (15x faster!)")
                    print("  regenerate *.java --parallel             # Regenerate Java files with parallel processing")
                    print("  regenerate config.yaml                   # Regenerate specific file")
                    print("  regenerate --parallel --concurrent 20    # Use 20 concurrent requests")
            
            elif command.lower() == 'help':
                print("\n🎯 Smart Commands:")
                print("• show filename.py     - Show summary for specific file")
                print("• show partial_name    - Find files with fuzzy matching")  
                print("• ls directory/        - List files in a directory")
                print("• find *.java          - Find files by pattern")
                print("• search <query>       - Search file content")
                print("• recent               - Show recently modified files")
                print("• list                 - Show all files")
                print("• regenerate           - Regenerate summaries")
                print("• exit                 - Quit")
            
            else:
                print("Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def show_file_summary(metadatas, documents, file_pattern):
    """Show summary for a specific file with smart matching"""
    if not file_pattern:
        print("Usage: show <filename or pattern>")
        print("Examples:")
        print("  show main.py")
        print("  show src/components/Button.tsx") 
        print("  show Button")
        return
    
    # Try exact match first
    exact_matches = []
    for i, meta in enumerate(metadatas):
        file_path = meta['file_path']
        if file_path == file_pattern or file_path.endswith('/' + file_pattern):
            exact_matches.append((i, meta, documents[i]))
    
    if exact_matches:
        if len(exact_matches) == 1:
            i, meta, doc = exact_matches[0]
            display_detailed_summary(meta, doc, i + 1)
        else:
            print(f"\n🎯 Found {len(exact_matches)} exact matches:")
            for idx, (i, meta, doc) in enumerate(exact_matches, 1):
                print(f"{idx}. {meta['file_path']}")
            
            try:
                choice = input("\nSelect file (1-{}): ".format(len(exact_matches))).strip()
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(exact_matches):
                    i, meta, doc = exact_matches[choice_idx]
                    display_detailed_summary(meta, doc, i + 1)
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
        return
    
    # Try fuzzy matching
    all_file_paths = [meta['file_path'] for meta in metadatas]
    all_basenames = [os.path.basename(path) for path in all_file_paths]
    
    # Match against basenames first
    basename_matches = get_close_matches(file_pattern, all_basenames, n=10, cutoff=0.3)
    
    # Also try partial matching
    partial_matches = []
    file_pattern_lower = file_pattern.lower()
    for file_path in all_file_paths:
        if file_pattern_lower in file_path.lower() or file_pattern_lower in os.path.basename(file_path).lower():
            partial_matches.append(file_path)
    
    # Combine and deduplicate matches
    all_matches = []
    for basename_match in basename_matches:
        for i, meta in enumerate(metadatas):
            if os.path.basename(meta['file_path']) == basename_match:
                all_matches.append((i, meta, documents[i]))
                break
    
    for partial_match in partial_matches:
        for i, meta in enumerate(metadatas):
            if meta['file_path'] == partial_match:
                # Avoid duplicates
                if not any(existing_meta['file_path'] == partial_match for _, existing_meta, _ in all_matches):
                    all_matches.append((i, meta, documents[i]))
                break
    
    if not all_matches:
        print(f"❌ No files found matching '{file_pattern}'")
        
        # Suggest similar files
        suggestions = get_close_matches(file_pattern, all_basenames, n=5, cutoff=0.1)
        if suggestions:
            print("\n💡 Did you mean one of these?")
            for suggestion in suggestions:
                for meta in metadatas:
                    if os.path.basename(meta['file_path']) == suggestion:
                        print(f"   • {meta['file_path']}")
                        break
        return
    
    if len(all_matches) == 1:
        i, meta, doc = all_matches[0]
        display_detailed_summary(meta, doc, i + 1)
    else:
        print(f"\n🎯 Found {len(all_matches)} matches for '{file_pattern}':")
        for idx, (i, meta, doc) in enumerate(all_matches, 1):
            print(f"{idx:2d}. {meta['file_path']} ({meta['file_type']}, {meta['language']})")
        
        try:
            choice = input(f"\nSelect file (1-{len(all_matches)}) or 'all' to see all: ").strip()
            if choice.lower() == 'all':
                for i, meta, doc in all_matches:
                    print("\n" + "="*80)
                    display_detailed_summary(meta, doc, i + 1)
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(all_matches):
                    i, meta, doc = all_matches[choice_idx]
                    display_detailed_summary(meta, doc, i + 1)
                else:
                    print("Invalid selection.")
        except ValueError:
            print("Invalid input.")

def list_directory_files(metadatas, dir_path):
    """List files in a specific directory"""
    if not dir_path:
        # Show root directories
        directories = set()
        files_in_root = []
        
        for meta in metadatas:
            file_path = meta['file_path']
            if '/' in file_path:
                root_dir = file_path.split('/')[0]
                directories.add(root_dir)
            else:
                files_in_root.append(meta)
        
        print(f"\n📁 Root directories:")
        for directory in sorted(directories):
            print(f"   📂 {directory}/")
        
        if files_in_root:
            print(f"\n📄 Files in root:")
            for meta in files_in_root:
                print(f"   📄 {meta['file_path']} ({meta['file_type']}, {meta['language']})")
        return
    
    # Normalize directory path
    dir_path = dir_path.rstrip('/')
    
    matching_files = []
    subdirectories = set()
    
    for meta in metadatas:
        file_path = meta['file_path']
        
        # Check if file is in the specified directory
        if file_path.startswith(dir_path + '/'):
            relative_path = file_path[len(dir_path) + 1:]
            
            if '/' in relative_path:
                # File is in a subdirectory
                subdir = relative_path.split('/')[0]
                subdirectories.add(subdir)
            else:
                # File is directly in this directory
                matching_files.append(meta)
    
    if not matching_files and not subdirectories:
        print(f"❌ No files found in directory '{dir_path}'")
        
        # Suggest similar directories
        all_dirs = set()
        for meta in metadatas:
            parts = meta['file_path'].split('/')
            for i in range(1, len(parts)):
                all_dirs.add('/'.join(parts[:i]))
        
        suggestions = get_close_matches(dir_path, list(all_dirs), n=5, cutoff=0.3)
        if suggestions:
            print("\n💡 Did you mean one of these directories?")
            for suggestion in suggestions:
                print(f"   • {suggestion}/")
        return
    
    print(f"\n📁 Contents of '{dir_path}/':")
    
    if subdirectories:
        print(f"\n📂 Subdirectories:")
        for subdir in sorted(subdirectories):
            print(f"   📂 {subdir}/")
    
    if matching_files:
        print(f"\n📄 Files ({len(matching_files)}):")
        for meta in sorted(matching_files, key=lambda x: x['file_path']):
            filename = os.path.basename(meta['file_path'])
            print(f"   📄 {filename} ({meta['file_type']}, {meta['language']}, {meta['line_count']} lines)")

def find_files_by_pattern(metadatas, documents, pattern):
    """Find files matching a glob pattern"""
    if not pattern:
        print("Usage: find <pattern>")
        print("Examples:")
        print("  find *.py")
        print("  find src/*.tsx")
        print("  find *test*")
        return
    
    matching_files = []
    for i, meta in enumerate(metadatas):
        file_path = meta['file_path']
        filename = os.path.basename(file_path)
        
        # Check both full path and filename
        if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(filename, pattern):
            matching_files.append((i, meta, documents[i]))
    
    if not matching_files:
        print(f"❌ No files found matching pattern '{pattern}'")
        return
    
    print(f"\n🔍 Found {len(matching_files)} files matching '{pattern}':")
    for idx, (i, meta, doc) in enumerate(matching_files, 1):
        print(f"{idx:2d}. {meta['file_path']} ({meta['file_type']}, {meta['language']}, {meta['line_count']} lines)")
        print(f"     Purpose: {meta['purpose']}")
    
    if len(matching_files) <= 5:
        show_all = input(f"\nShow detailed summaries? (y/N): ").lower() == 'y'
        if show_all:
            for i, meta, doc in matching_files:
                print("\n" + "="*80)
                display_detailed_summary(meta, doc, i + 1)

def show_recent_files(metadatas, documents):
    """Show recently modified files"""
    files_with_mtime = []
    
    for i, meta in enumerate(metadatas):
        file_path = meta['file_path']
        if os.path.exists(file_path):
            try:
                mtime = os.path.getmtime(file_path)
                files_with_mtime.append((mtime, i, meta, documents[i]))
            except OSError:
                pass
    
    if not files_with_mtime:
        print("❌ No accessible files found")
        return
    
    # Sort by modification time (most recent first)
    files_with_mtime.sort(reverse=True)
    recent_files = files_with_mtime[:10]  # Show top 10
    
    print(f"\n🕒 Recently modified files:")
    import datetime
    for idx, (mtime, i, meta, doc) in enumerate(recent_files, 1):
        mod_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        print(f"{idx:2d}. {meta['file_path']} ({mod_time})")
        print(f"     {meta['file_type']}, {meta['language']}, {meta['line_count']} lines")

def display_detailed_summary(meta, doc, file_number):
    """Display detailed summary for a file"""
    print(f"\n📄 File Summary:")
    print(f"File: {meta['file_path']}")
    print(f"Language: {meta['language']}")
    print(f"Type: {meta['file_type']}")
    print(f"Lines: {meta['line_count']}")
    print(f"Complexity: {meta['complexity_score']}")
    print(f"Purpose: {meta['purpose']}")
    print(f"\n🤖 AI-Generated Summary:")
    print("-" * 50)
    print(doc)

def regenerate_all_summaries(indexer, use_parallel: bool = False, concurrent: int = 15):
    """Regenerate all summaries using AI-powered analysis"""
    if not indexer:
        print("❌ No indexer available")
        return 0
    
    if use_parallel:
        print(f"🚀 Using PARALLEL processing with {concurrent} concurrent requests")
    else:
        print("🐌 Using SEQUENTIAL processing")
        
    try:
        # Get all files in the database
        collection = indexer.client.get_collection(indexer.collection_name)
        all_data = collection.get()
        
        if use_parallel:
            # Collect all file data for parallel processing
            file_data = []
            valid_files = []
            
            for metadata in all_data['metadatas']:
                file_path_str = metadata['file_path']
                try:
                    if os.path.exists(file_path_str):
                        with open(file_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        file_data.append((file_path_str, content))
                        valid_files.append(file_path_str)
                    else:
                        print(f"  ⚠️ File not found: {file_path_str}")
                except Exception as e:
                    print(f"  ❌ Error reading {file_path_str}: {e}")
            
            if file_data:
                print(f"📊 Processing {len(file_data)} files with parallel AI...")
                
                # Clear the collection first
                if all_data['ids']:
                    collection.delete(ids=all_data['ids'])
                
                # Create new indexer for parallel processing
                from code_indexer import CodeIndexer
                new_indexer = CodeIndexer(max_concurrent=concurrent)
                
                # Process all files in parallel
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    ai_summaries = loop.run_until_complete(
                        new_indexer.generate_all_summaries_parallel(file_data)
                    )
                    
                    # Create FileSummary objects and add to database
                    all_summaries = []
                    for file_path, content in file_data:
                        ai_summary = ai_summaries.get(file_path, "No AI summary available")
                        file_summary = new_indexer.create_file_summary_with_ai_insights(
                            file_path, content, ai_summary
                        )
                        all_summaries.append(file_summary)
                    
                    # Add to vector database
                    new_indexer.add_summaries_to_vector_db(all_summaries)
                    return len(all_summaries)
                    
                finally:
                    loop.close()
            
        else:
            # Sequential processing (original method)
            regenerated_count = 0
            
            for i, metadata in enumerate(all_data['metadatas']):
                file_path_str = metadata['file_path']
                print(f"📄 Processing ({i+1}/{len(all_data['metadatas'])}): {file_path_str}")
                
                try:
                    if os.path.exists(file_path_str):
                        # Re-index this file with AI summaries
                        indexer.index_file(file_path_str, force_reindex=True)
                        regenerated_count += 1
                        print(f"  ✅ Regenerated summary")
                    else:
                        print(f"  ⚠️ File not found, skipping")
                        
                except Exception as e:
                    print(f"  ❌ Error processing file: {e}")
                    
            return regenerated_count
        
    except Exception as e:
        print(f"❌ Error regenerating summaries: {e}")
        return 0

def regenerate_matching_summaries(indexer, pattern: str, use_parallel: bool = False, concurrent: int = 15):
    """Regenerate summaries for files matching a pattern"""
    import fnmatch
    
    if not indexer:
        print("❌ No indexer available")
        return 0
        
    try:
        # Get all files in the database
        collection = indexer.client.get_collection(indexer.collection_name)
        all_data = collection.get()
        
        matching_files = []
        for metadata in all_data['metadatas']:
            file_path = metadata['file_path']
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):
                matching_files.append(file_path)
        
        print(f"Found {len(matching_files)} files matching pattern '{pattern}'")
        
        if use_parallel and matching_files:
            print(f"🚀 Using PARALLEL processing with {concurrent} concurrent requests")
            
            # Collect file data for parallel processing
            file_data = []
            valid_files = []
            
            for file_path in matching_files:
                try:
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        file_data.append((file_path, content))
                        valid_files.append(file_path)
                    else:
                        print(f"  ⚠️ File not found: {file_path}")
                except Exception as e:
                    print(f"  ❌ Error reading {file_path}: {e}")
            
            if file_data:
                # Create new indexer for parallel processing
                from code_indexer import CodeIndexer
                new_indexer = CodeIndexer(max_concurrent=concurrent)
                
                # Process matching files in parallel
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    ai_summaries = loop.run_until_complete(
                        new_indexer.generate_all_summaries_parallel(file_data)
                    )
                    
                    # Update summaries in the original indexer
                    for file_path in valid_files:
                        # Remove old summary
                        try:
                            existing = indexer.collection.get(where={"file_path": file_path})
                            if existing['ids']:
                                indexer.collection.delete(ids=existing['ids'])
                        except:
                            pass
                        
                        # Add new summary
                        content = next(content for fp, content in file_data if fp == file_path)
                        ai_summary = ai_summaries.get(file_path, "No AI summary available")
                        file_summary = new_indexer.create_file_summary_with_ai_insights(
                            file_path, content, ai_summary
                        )
                        
                        # Add to original indexer's collection
                        import hashlib
                        summary_id = f"file_{hashlib.md5(file_path.encode()).hexdigest()}"
                        indexer.collection.add(
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
                    
                    return len(valid_files)
                    
                finally:
                    loop.close()
        
        else:
            # Sequential processing (original method)
            print("🐌 Using SEQUENTIAL processing")
            regenerated_count = 0
            for i, file_path in enumerate(matching_files):
                print(f"📄 Processing ({i+1}/{len(matching_files)}): {file_path}")
                
                try:
                    if os.path.exists(file_path):
                        # Re-index this file with AI summaries
                        indexer.index_file(file_path, force_reindex=True)
                        regenerated_count += 1
                        print(f"  ✅ Regenerated summary")
                    else:
                        print(f"  ⚠️ File not found, skipping")
                        
                except Exception as e:
                    print(f"  ❌ Error processing file: {e}")
                    
            return regenerated_count
        
    except Exception as e:
        print(f"❌ Error regenerating summaries: {e}")
        return 0

if __name__ == "__main__":
    main() 