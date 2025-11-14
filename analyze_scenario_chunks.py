#!/usr/bin/env python3
"""Analyze chunks from all scenario files"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add locobench to path
sys.path.insert(0, str(Path(__file__).parent))

from locobench.tools.file_chunking import chunk_file_smart
from locobench.tools.chunk_analysis import analyze_chunk_relevance


def analyze_scenario_chunks(
    scenarios_dir: str = "data/output/scenarios",
    base_path: str = "/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated",
    max_files_per_scenario: int = 5
) -> Dict[str, Dict]:
    """
    Analyze chunks for all scenario files.
    
    Args:
        scenarios_dir: Directory containing scenario JSON files
        base_path: Base path for generated projects
        max_files_per_scenario: Maximum files to analyze per scenario
    
    Returns:
        Dictionary mapping scenario_id to analysis results
    """
    scenarios_path = Path(scenarios_dir)
    if not scenarios_path.exists():
        print(f"Error: Scenarios directory not found: {scenarios_dir}")
        return {}
    
    results = {}
    scenario_files = list(scenarios_path.glob("*.json"))
    
    print(f"Found {len(scenario_files)} scenario files")
    
    for scenario_file in scenario_files:
        scenario_id = scenario_file.stem
        print(f"\nAnalyzing scenario: {scenario_id}")
        
        try:
            # Read scenario file
            scenario_data = json.loads(scenario_file.read_text(encoding='utf-8'))
            
            # Extract project name
            task_categories = [
                'architectural_understanding', 'cross_file_refactoring', 'feature_implementation',
                'bug_investigation', 'multi_session_development', 'code_comprehension',
                'integration_testing', 'security_analysis'
            ]
            
            project_name = None
            for category in task_categories:
                if f'_{category}' in scenario_id:
                    project_name = scenario_id.split(f'_{category}')[0]
                    break
            
            if not project_name:
                print(f"  ⚠️  Could not extract project name from {scenario_id}")
                continue
            
            # Get context files
            context_files = scenario_data.get('context_files', [])
            if isinstance(context_files, dict):
                context_files = list(context_files.keys())
            elif not isinstance(context_files, list):
                print(f"  ⚠️  No context_files found")
                continue
            
            if not context_files:
                print(f"  ⚠️  Empty context_files list")
                continue
            
            task_prompt = scenario_data.get('task_prompt', '')
            
            # Build full paths
            base_dir = Path(base_path) / project_name
            full_paths = []
            
            for rel_path in context_files[:max_files_per_scenario]:
                normalized = rel_path.replace('//', '/').replace('\\', '/').lstrip('/')
                full_path = base_dir / normalized
                if full_path.exists():
                    full_paths.append(str(full_path))
            
            if not full_paths:
                print(f"  ⚠️  No valid files found")
                continue
            
            # Analyze chunks for each file
            file_analyses = []
            total_chunks = 0
            
            for file_path in full_paths:
                try:
                    path = Path(file_path)
                    content = path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Chunk the file
                    chunks = chunk_file_smart(content, max_chunk_size=2000)
                    total_chunks += len(chunks)
                    
                    # Analyze relevance of chunks
                    chunk_scores = []
                    for chunk in chunks:
                        score = analyze_chunk_relevance(chunk['content'], task_prompt)
                        chunk_scores.append({
                            'chunk_index': chunk['chunk_index'],
                            'score': score,
                            'start_line': chunk.get('start_line', 0),
                            'end_line': chunk.get('end_line', 0),
                            'size': len(chunk['content'])
                        })
                    
                    # Sort by score
                    chunk_scores.sort(key=lambda x: x['score'], reverse=True)
                    
                    file_analyses.append({
                        'file_path': str(path),
                        'file_name': path.name,
                        'total_chunks': len(chunks),
                        'top_chunks': chunk_scores[:3],  # Top 3 chunks
                        'max_score': max([c['score'] for c in chunk_scores]) if chunk_scores else 0.0
                    })
                    
                except Exception as e:
                    print(f"  ⚠️  Error analyzing {file_path}: {e}")
                    continue
            
            # Find best file
            if file_analyses:
                best_file = max(file_analyses, key=lambda x: x['max_score'])
                results[scenario_id] = {
                    'project_name': project_name,
                    'total_files_analyzed': len(file_analyses),
                    'total_chunks': total_chunks,
                    'best_file': best_file['file_name'],
                    'best_file_score': best_file['max_score'],
                    'file_analyses': file_analyses
                }
                
                print(f"  ✓ Analyzed {len(file_analyses)} files, {total_chunks} total chunks")
                print(f"  ✓ Best file: {best_file['file_name']} (score: {best_file['max_score']:.3f})")
            else:
                print(f"  ⚠️  No files analyzed")
                
        except Exception as e:
            print(f"  ✗ Error processing {scenario_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze chunks from scenario files")
    parser.add_argument("--scenarios-dir", default="data/output/scenarios",
                       help="Directory containing scenario JSON files")
    parser.add_argument("--base-path", 
                       default="/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/data/generated",
                       help="Base path for generated projects")
    parser.add_argument("--max-files", type=int, default=5,
                       help="Maximum files to analyze per scenario")
    parser.add_argument("--output", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    results = analyze_scenario_chunks(
        scenarios_dir=args.scenarios_dir,
        base_path=args.base_path,
        max_files_per_scenario=args.max_files
    )
    
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n✓ Results saved to {output_path}")
    else:
        print(f"\n✓ Analysis complete. Processed {len(results)} scenarios.")
        print("\nSummary:")
        for scenario_id, data in results.items():
            print(f"  {scenario_id}: {data['total_chunks']} chunks, best file: {data['best_file']}")
