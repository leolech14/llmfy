#!/usr/bin/env python3
"""
📝 Markdown Converter - Convert plain text documents to properly formatted markdown

This preprocessor identifies structure in plain text documents and converts them
to markdown format for better quality scoring and processing.
"""

import re
from pathlib import Path
from typing import List, Tuple
from rich.console import Console

console = Console()

class MarkdownConverter:
    """Convert plain text documents to markdown format"""
    
    def __init__(self):
        self.section_patterns = [
            # Numbered sections like "1. Section Title"
            (r'^(\d+)\.\s+([A-Z][^.!?\n]+)$', r'## \1. \2'),
            # All-caps lines that are likely headers
            (r'^([A-Z][A-Z\s]+)$', r'### \1'),
            # Lines ending with colon that introduce content
            (r'^([A-Z][^:]+):$', r'### \1:'),
        ]
        
    def convert_to_markdown(self, text: str) -> str:
        """Convert plain text to markdown format"""
        lines = text.split('\n')
        converted_lines = []
        in_paragraph = False
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                converted_lines.append('')
                in_paragraph = False
                continue
            
            # Check if it's a title (first line or after empty line, followed by shorter line)
            if i == 0 or (i > 0 and not lines[i-1].strip()):
                if i < len(lines) - 1 and len(line) > 50 and len(lines[i+1]) < 50:
                    converted_lines.append(f"# {line}")
                    in_paragraph = False
                    continue
            
            # Check for section patterns
            converted = False
            for pattern, replacement in self.section_patterns:
                if re.match(pattern, line):
                    converted_lines.append(re.sub(pattern, replacement, line))
                    converted = True
                    in_paragraph = False
                    break
            
            if converted:
                continue
            
            # Add line to output
            converted_lines.append(line)
            
        # Join lines and ensure proper paragraph spacing
        result = '\n'.join(converted_lines)
        
        # Add code block formatting for inline URLs/references
        result = re.sub(r'(\w+\.\w+)(?=\s|$)', r'`\1`', result)
        
        # Format examples with proper markdown
        result = re.sub(r'For example[,:]?\s*', r'\n**Example:** ', result, flags=re.IGNORECASE)
        result = re.sub(r'For instance[,:]?\s*', r'\n**Example:** ', result, flags=re.IGNORECASE)
        result = re.sub(r'e\.g\.\s*', r'*e.g.,* ', result)
        
        # Add emphasis to key terms
        result = self._emphasize_key_terms(result)
        
        # Ensure proper spacing between sections
        result = re.sub(r'\n(#{1,3}\s+)', r'\n\n\1', result)
        
        return result
    
    def _emphasize_key_terms(self, text: str) -> str:
        """Add emphasis to important terms and concepts"""
        # Key design terms
        design_terms = [
            'Gestalt', 'proximity', 'similarity', 'closure', 'figure-ground',
            'typography', 'vertical rhythm', 'modular scale', 'whitespace',
            'hierarchy', 'consistency', 'accessibility', 'semantic',
            'responsive', 'pixel-perfect', 'user experience', 'interface'
        ]
        
        for term in design_terms:
            # Only emphasize if not already in a header or emphasis
            pattern = rf'(?<!#\s)(?<!\*)\b({term})\b(?!\*)'
            text = re.sub(pattern, r'**\1**', text, flags=re.IGNORECASE)
        
        return text
    
    def process_file(self, input_path: Path, output_path: Path = None) -> Path:
        """Process a file and convert to markdown"""
        
        console.print(f"[blue]Converting {input_path.name} to markdown...[/blue]")
        
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert to markdown
        markdown_content = self.convert_to_markdown(content)
        
        # Determine output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_formatted.md"
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        console.print(f"[green]✅ Converted to markdown: {output_path.name}[/green]")
        
        return output_path


def main():
    """CLI for markdown converter"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert plain text documents to markdown format"
    )
    parser.add_argument('input', help='Input file path')
    parser.add_argument('-o', '--output', help='Output file path')
    
    args = parser.parse_args()
    
    converter = MarkdownConverter()
    converter.process_file(Path(args.input), Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
