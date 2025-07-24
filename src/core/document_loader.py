import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import tempfile
import subprocess
import shutil

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain.schema import Document
import markdown
from rich.console import Console
from rich.progress import track

from .config import Config

console = Console()

class DocumentLoader:
    def __init__(self):
        self.config = Config()
        self.loaded_documents = []
        self.metadata_cache = self._load_metadata_cache()
    
    def _load_metadata_cache(self) -> Dict[str, Any]:
        """Load metadata cache to track processed files"""
        cache_file = self.config.DATA_DIR / "metadata_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata_cache(self):
        """Save metadata cache"""
        cache_file = self.config.DATA_DIR / "metadata_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.metadata_cache, f, indent=2)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file content"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _is_file_processed(self, file_path: Path) -> bool:
        """Check if file has already been processed"""
        file_hash = self._get_file_hash(file_path)
        file_key = str(file_path)
        
        if file_key in self.metadata_cache:
            return self.metadata_cache[file_key]['hash'] == file_hash
        return False
    
    def _get_markdown_loader(self):
        """Create a custom markdown loader class"""
        class MarkdownLoader:
            def __init__(self, file_path: str):
                self.file_path = file_path
            
            def load(self) -> List[Document]:
                """Load markdown file and convert to HTML for better structure preservation"""
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert markdown to HTML for better parsing
                md = markdown.Markdown(extensions=['extra', 'codehilite', 'toc'])
                html_content = md.convert(content)
                
                # Create document with both raw and HTML content
                return [Document(
                    page_content=content,
                    metadata={
                        'source': self.file_path,
                        'html_content': html_content,
                        'format': 'markdown'
                    }
                )]
        
        return MarkdownLoader

    def _get_loader_for_file(self, file_path: Path):
        """Get appropriate loader based on file extension"""
        extension = file_path.suffix.lower()
        
        loader_map = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': self._get_markdown_loader(),
            '.markdown': self._get_markdown_loader(),
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
        }
        
        # For code files, use TextLoader
        if extension in {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', 
                        '.h', '.hpp', '.cs', '.rb', '.go', '.rs', '.swift', '.html', 
                        '.htm', '.xml', '.json', '.yaml', '.yml'}:
            return TextLoader
        
        return loader_map.get(extension, TextLoader)
    
    def load_file(self, file_path: Path, force: bool = False) -> List[Document]:
        """Load a single file
        
        Args:
            file_path: Path to the file to load
            force: If True, force reprocessing even if file was already processed
        """
        if not file_path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            return []
        
        if not force and self._is_file_processed(file_path):
            console.print(f"[yellow]Skipping already processed file: {file_path.name}[/yellow]")
            console.print(f"[dim]Use --force to reprocess[/dim]")
            return []
        
        try:
            loader_class = self._get_loader_for_file(file_path)
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'filename': file_path.name,
                    'file_type': file_path.suffix.lower(),
                    'load_timestamp': datetime.now().isoformat(),
                    'file_hash': self._get_file_hash(file_path)
                })
            
            # Update metadata cache
            self.metadata_cache[str(file_path)] = {
                'hash': self._get_file_hash(file_path),
                'processed_at': datetime.now().isoformat(),
                'num_documents': len(documents)
            }
            self._save_metadata_cache()
            
            console.print(f"[green]Successfully loaded: {file_path.name} ({len(documents)} documents)[/green]")
            return documents
            
        except Exception as e:
            console.print(f"[red]Error loading {file_path.name}: {str(e)}[/red]")
            return []
    
    def load_directory(self, directory_path: Path, recursive: bool = True, force: bool = False) -> List[Document]:
        """Load all supported files from a directory
        
        Args:
            directory_path: Path to directory to load
            recursive: If True, load files recursively
            force: If True, force reprocessing of all files
        """
        documents = []
        pattern = "**/*" if recursive else "*"
        
        files = [f for f in directory_path.glob(pattern) 
                if f.is_file() and f.suffix.lower() in self.config.SUPPORTED_EXTENSIONS]
        
        console.print(f"[blue]Found {len(files)} files to process[/blue]")
        
        for file_path in track(files, description="Loading files..."):
            docs = self.load_file(file_path, force=force)
            documents.extend(docs)
        
        return documents
    
    def load_github_repo(self, repo_url: str, branch: str = "main", 
                        file_filter: Optional[List[str]] = None) -> List[Document]:
        """Load documents from a GitHub repository"""
        import tempfile
        import subprocess
        import shutil
        
        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="github_repo_")
            console.print(f"[blue]Cloning repository: {repo_url}[/blue]")
            
            # Try cloning with different branch options
            clone_successful = False
            for branch_option in [branch, "master", None]:
                if branch_option:
                    cmd = ["git", "clone", "--depth", "1", "--branch", branch_option, repo_url, temp_dir]
                else:
                    cmd = ["git", "clone", "--depth", "1", repo_url, temp_dir]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    clone_successful = True
                    actual_branch = branch_option or "default"
                    console.print(f"[green]Successfully cloned repository (branch: {actual_branch})[/green]")
                    break
            
            if not clone_successful:
                raise Exception(f"Failed to clone repository: {result.stderr}")
            
            # Load documents from the cloned directory
            temp_path = Path(temp_dir)
            documents = self.load_directory(temp_path, recursive=True)
            
            # Add GitHub-specific metadata
            for doc in documents:
                doc.metadata.update({
                    'source_type': 'github',
                    'repo_url': repo_url,
                    'branch': actual_branch,
                    'load_timestamp': datetime.now().isoformat()
                })
            
            console.print(f"[green]Successfully loaded {len(documents)} documents from {repo_url}[/green]")
            return documents
            
        except Exception as e:
            console.print(f"[red]Error loading GitHub repo: {str(e)}[/red]")
            return []
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                console.print(f"[dim]Cleaned up temporary directory[/dim]")
    
    def _should_include_file(self, file_path: str, file_filter: Optional[List[str]] = None) -> bool:
        """Check if file should be included based on filters"""
        if file_filter:
            return any(pattern in file_path for pattern in file_filter)
        
        # Check if file has supported extension
        path = Path(file_path)
        return path.suffix.lower() in self.config.SUPPORTED_EXTENSIONS
    
    def load_from_inbox(self) -> List[Document]:
        """Load all new documents from the inbox directory"""
        console.print("[bold blue]Loading documents from inbox...[/bold blue]")
        documents = self.load_directory(self.config.INBOX_DIR, recursive=True)
        self.loaded_documents.extend(documents)
        return documents
    
    def move_to_processed(self, file_path: Path):
        """Move a file from inbox to processed directory"""
        if file_path.exists():
            relative_path = file_path.relative_to(self.config.INBOX_DIR)
            dest_path = self.config.PROCESSED_DIR / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_path.rename(dest_path)
            console.print(f"[green]Moved {file_path.name} to processed[/green]")
