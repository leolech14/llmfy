#!/usr/bin/env python3
"""
🚀 llmfy AI Library - Quick Start Script

One command to set up your quality-first AI knowledge system.
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
import time

console = Console()

class llmfyQuickStart:
    def __init__(self):
        self.nexus_root = Path(__file__).parent
        self.venv_path = self.nexus_root / "venv"
        self.requirements = [
            "langchain>=0.0.300",
            "chromadb>=0.4.0",
            "openai>=0.27.0",
            "sentence-transformers",
            "python-dotenv>=1.0.0",
            "rich>=13.0.0",
            "pyyaml>=6.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0"
        ]
        
    def welcome(self):
        """Display welcome message"""
        console.print(Panel.fit(
            "[bold cyan]🧠 Welcome to llmfy AI Library System[/bold cyan]\n\n"
            "[white]The quality-first knowledge management system[/white]\n"
            "[dim]Every chunk must score 9.5/10 or higher[/dim]",
            border_style="cyan"
        ))
        console.print()
        
    def check_environment(self):
        """Check Python version and environment"""
        console.print("[bold]🔍 Checking environment...[/bold]")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            console.print("[red]❌ Python 3.8+ required[/red]")
            return False
        console.print(f"[green]✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}[/green]")
        
        # Check if running in existing venv
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            console.print("[yellow]⚠️  Already in virtual environment[/yellow]")
            if not Confirm.ask("Continue with current environment?"):
                return False
                
        return True
        
    def create_virtual_environment(self):
        """Create and activate virtual environment"""
        if self.venv_path.exists():
            console.print("[yellow]⚠️  Virtual environment already exists[/yellow]")
            if Confirm.ask("Recreate virtual environment?"):
                import shutil
                shutil.rmtree(self.venv_path)
            else:
                return True
                
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating virtual environment...", total=None)
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            progress.remove_task(task)
            
        console.print("[green]✅ Virtual environment created[/green]")
        return True
        
    def install_dependencies(self):
        """Install required packages"""
        console.print("\n[bold]📦 Installing dependencies...[/bold]")
        
        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = self.venv_path / "Scripts" / "pip"
        else:  # Unix-like
            pip_path = self.venv_path / "bin" / "pip"
            
        # Upgrade pip first
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], capture_output=True)
        
        # Install requirements
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for req in self.requirements:
                task = progress.add_task(f"Installing {req.split('>')[0]}...", total=None)
                result = subprocess.run(
                    [str(pip_path), "install", req],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    console.print(f"[red]❌ Failed to install {req}[/red]")
                    console.print(f"[dim]{result.stderr}[/dim]")
                else:
                    console.print(f"[green]✅ {req.split('>')[0]} installed[/green]")
                progress.remove_task(task)
                
    def create_config_files(self):
        """Create default configuration files"""
        console.print("\n[bold]⚙️  Creating configuration files...[/bold]")
        
        # Main config
        main_config = """# llmfy AI Library Configuration

environment: development

# Embedding Configuration
embeddings:
  development:
    provider: local
    model: all-MiniLM-L6-v2
  production:
    provider: hybrid
    local_model: all-MiniLM-L6-v2
    cloud_model: text-embedding-3-small
    routing: intelligent

# Storage Configuration  
storage:
  development:
    provider: chromadb
    path: ./data/chromadb
  production:
    provider: hybrid
    local: chromadb
    cloud: pinecone
    
# Quality Configuration
quality:
  threshold: 9.5
  auto_enhance: true
  enforce_threshold: true
  
# Processing Configuration
processing:
  chunk_size: 1500
  chunk_overlap: 200
  batch_size: 100
"""
        
        # Quality rules
        quality_rules = """# Quality Scoring Rules

# Weights for each dimension (must sum to 1.0)
weights:
  self_contained: 0.20
  definitions: 0.15
  examples: 0.20
  structure: 0.10
  relationships: 0.15
  clarity: 0.10
  completeness: 0.10

# Minimum scores for each dimension
minimum_scores:
  self_contained: 8.0
  definitions: 7.0
  examples: 8.0
  structure: 7.0
  relationships: 7.0
  clarity: 8.0
  completeness: 8.0

# Enhancement rules
enhancements:
  add_context: true
  define_terms: true
  provide_examples: true
  improve_structure: true
  link_concepts: true
"""
        
        # Create config directory
        config_dir = self.nexus_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Write configs
        (config_dir / "nexus_config.yaml").write_text(main_config)
        (config_dir / "quality_rules.yaml").write_text(quality_rules)
        
        console.print("[green]✅ Configuration files created[/green]")
        
    def create_env_file(self):
        """Create .env template"""
        env_template = """# llmfy AI Library Environment Variables

# OpenAI API (for production embeddings)
OPENAI_API_KEY=your_api_key_here

# Pinecone (for cloud storage)
PINECONE_API_KEY=your_api_key_here
PINECONE_ENV=us-east-1-aws
PINECONE_INDEX_NAME=nexus-knowledge

# Environment
NEXUS_ENV=development

# Quality Settings
QUALITY_THRESHOLD=9.5
AUTO_ENHANCE=true

# Cost Optimization
USE_CACHE=true
CACHE_EMBEDDINGS=true
MAX_MONTHLY_COST=100
"""
        
        env_path = self.nexus_root / ".env"
        if not env_path.exists():
            env_path.write_text(env_template)
            console.print("[green]✅ .env template created[/green]")
            console.print("[yellow]📝 Remember to add your API keys to .env[/yellow]")
        else:
            console.print("[yellow]⚠️  .env already exists[/yellow]")
            
    def test_installation(self):
        """Test the installation"""
        console.print("\n[bold]🧪 Testing installation...[/bold]")
        
        # Test imports
        test_script = """
import sys
sys.path.insert(0, '.')

try:
    from src.core.config import Config
    from src.quality.quality_scorer import QualityAnalyzer
    print("✅ Core imports successful")
    
    # Test quality scoring
    analyzer = QualityAnalyzer()
    test_text = "This is a test chunk to verify the quality scoring system is working."
    result = analyzer.analyze(test_text)
    print(f"✅ Quality scoring works: {result['overall_score']:.2f}/10")
    
    # Test configuration
    config = Config()
    print("✅ Configuration loaded")
    
    print("\n🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
"""
        
        # Write test script
        test_path = self.nexus_root / "_test_install.py"
        test_path.write_text(test_script)
        
        # Run test
        if os.name == 'nt':
            python_path = self.venv_path / "Scripts" / "python"
        else:
            python_path = self.venv_path / "bin" / "python"
            
        result = subprocess.run(
            [str(python_path), str(test_path)],
            capture_output=True,
            text=True,
            cwd=str(self.nexus_root)
        )
        
        console.print(result.stdout)
        if result.returncode != 0:
            console.print("[red]Installation test failed![/red]")
            console.print(result.stderr)
        
        # Clean up
        test_path.unlink()
        
    def display_next_steps(self):
        """Show next steps"""
        console.print("\n" + "="*60 + "\n")
        
        console.print(Panel(
            "[bold green]🎉 llmfy AI Library is ready![/bold green]\n\n"
            "[bold]Next steps:[/bold]\n\n"
            "1. Activate the virtual environment:\n"
            f"   [cyan]source {self.venv_path}/bin/activate[/cyan] (Unix)\n"
            f"   [cyan]{self.venv_path}\\Scripts\\activate[/cyan] (Windows)\n\n"
            "2. Add your API keys to .env (for production)\n\n"
            "3. Process your first document:\n"
            "   [cyan]python -m src.core.nexus_pipeline --input data/inbox/test.md[/cyan]\n\n"
            "4. Check quality scores:\n"
            "   [cyan]python nexus_validator.py data/processed/[/cyan]\n\n"
            "[dim]Remember: Quality first. Every chunk must be 9.5/10 or higher.[/dim]",
            border_style="green"
        ))
        
    def run(self):
        """Run the complete setup process"""
        self.welcome()
        
        if not self.check_environment():
            return
            
        try:
            self.create_virtual_environment()
            self.install_dependencies()
            self.create_config_files()
            self.create_env_file()
            self.test_installation()
            self.display_next_steps()
            
        except Exception as e:
            console.print(f"\n[red]❌ Setup failed: {e}[/red]")
            console.print("[yellow]Please check the error and try again[/yellow]")
            return

if __name__ == "__main__":
    quickstart = llmfyQuickStart()
    quickstart.run()
