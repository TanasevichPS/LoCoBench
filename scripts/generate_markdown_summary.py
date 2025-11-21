#!/usr/bin/env python3
"""
Standalone script to generate markdown summary from evaluation results.

This script reads evaluation results from intermediate_results/evaluation_incremental_results.json
and generates a comprehensive markdown summary report.

Usage:
    python scripts/generate_markdown_summary.py [--config-path config.yaml] [--output-file summary.md]
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import statistics

# Add parent directory to path to import locobench modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from locobench.core.config import Config
from locobench.core.task import TaskCategory, DifficultyLevel


@dataclass
class ModelEvaluationResult:
    """Results for a single model on a single scenario"""
    model_name: str
    scenario_id: str
    scenario_title: str
    task_category: str
    difficulty: str
    
    # Core scores (from ValidationResult) - 4 Evaluation Dimensions
    software_engineering_score: float    # 40% - Software Engineering Excellence (8 metrics)
    functional_correctness_score: float  # 30% - Functional Correctness (4 metrics)  
    code_quality_score: float           # 20% - Code Quality Assessment (3 metrics)
    longcontext_utilization_score: float # 10% - Long-Context Utilization (2 metrics)
    total_score: float
    
    # Additional metrics
    generation_time: float
    code_files_generated: int
    total_lines_generated: int
    parsing_success: bool
    prompt_length_chars: int
    
    # Solution code preservation
    solution_code: Dict[str, str]  # filename -> code content
    generated_files: List[str]     # list of filenames generated
    
    # Detailed breakdown
    detailed_results: Dict[str, Any]
    timestamp: str


@dataclass
class EvaluationSummary:
    """Summary statistics for model evaluation"""
    model_name: str
    total_scenarios: int
    completed_scenarios: int
    failed_scenarios: int
    
    avg_software_engineering_score: float
    avg_functional_correctness_score: float
    avg_code_quality_score: float
    avg_longcontext_utilization_score: float
    avg_total_score: float
    
    avg_generation_time: float
    total_evaluation_time: float
    parsing_success_rate: float
    
    category_results: Dict[str, Dict[str, Any]]
    difficulty_results: Dict[str, Dict[str, Any]]


class MarkdownSummaryGenerator:
    """Generate markdown summary from evaluation results"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_results_from_jsonl(self, results_file: Union[Path, List[Path]]) -> List[ModelEvaluationResult]:
        """Load evaluation results from JSONL file or multiple files"""
        # Handle both single file and list of files
        if isinstance(results_file, list):
            files = results_file
        else:
            files = [results_file]
        
        all_results = []
        for file_path in files:
            if not file_path.exists():
                print(f"⚠️  Results file not found: {file_path}")
                continue
            
            results = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            item = json.loads(line)
                            result = self._create_result_with_compatibility(item)
                            results.append(result)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Skipping corrupted line {line_num} in {file_path.name}: {e}")
                            continue
                        except Exception as e:
                            print(f"⚠️  Error processing line {line_num} in {file_path.name}: {e}")
                            continue
                
                print(f"✅ Loaded {len(results)} evaluation results from {file_path.name}")
                all_results.extend(results)
                
            except Exception as e:
                print(f"⚠️  Failed to load results file {file_path}: {e}")
                continue
        
        if not all_results:
            raise Exception(f"No results loaded from any file")
        
        print(f"📊 Total: {len(all_results)} evaluation results loaded")
        return all_results
    
    def _create_result_with_compatibility(self, item: Dict[str, Any]) -> ModelEvaluationResult:
        """Create ModelEvaluationResult with backward compatibility for old field names"""
        # Map old field names to new field names
        field_mapping = {
            'functional_score': 'functional_correctness_score',
            'agent_metrics_score': 'software_engineering_score', 
            'longcontext_metrics_score': 'software_engineering_score',
            'quality_score': 'code_quality_score',
            'style_score': 'longcontext_utilization_score'
        }
        
        # Create a copy of the item to avoid modifying original
        mapped_item = item.copy()
        
        # Map old field names to new ones
        for old_field, new_field in field_mapping.items():
            if old_field in mapped_item and new_field not in mapped_item:
                mapped_item[new_field] = mapped_item.pop(old_field)
        
        # Set default values for any missing new fields
        default_values = {
            'software_engineering_score': 0.0,
            'functional_correctness_score': 0.0,
            'code_quality_score': 0.0,
            'longcontext_utilization_score': 0.0,
            'prompt_length_chars': 0,
            'solution_code': {},
            'generated_files': [],
            'detailed_results': {}
        }
        
        for field, default_value in default_values.items():
            if field not in mapped_item:
                mapped_item[field] = default_value
        
        return ModelEvaluationResult(**mapped_item)
    
    def generate_evaluation_summary(self, results: Dict[str, List[ModelEvaluationResult]]) -> Dict[str, EvaluationSummary]:
        """Generate comprehensive evaluation summaries"""
        
        summaries = {}
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            # Calculate averages
            total_scenarios = len(model_results)
            completed_scenarios = len([r for r in model_results if r.parsing_success])
            failed_scenarios = total_scenarios - completed_scenarios
            
            avg_software_engineering = sum(r.software_engineering_score for r in model_results) / total_scenarios
            avg_functional_correctness = sum(r.functional_correctness_score for r in model_results) / total_scenarios
            avg_code_quality = sum(r.code_quality_score for r in model_results) / total_scenarios
            avg_longcontext_utilization = sum(r.longcontext_utilization_score for r in model_results) / total_scenarios
            avg_total = sum(r.total_score for r in model_results) / total_scenarios
            
            avg_generation_time = sum(r.generation_time for r in model_results) / total_scenarios
            parsing_success_rate = completed_scenarios / total_scenarios
            
            # Category breakdown
            category_results = {}
            for category in TaskCategory:
                category_name = category.value
                category_scores = [r for r in model_results if r.task_category == category_name]
                if category_scores:
                    category_results[category_name] = {
                        'count': len(category_scores),
                        'avg_total_score': sum(r.total_score for r in category_scores) / len(category_scores),
                        'avg_software_engineering': sum(r.software_engineering_score for r in category_scores) / len(category_scores),
                        'avg_functional_correctness': sum(r.functional_correctness_score for r in category_scores) / len(category_scores),
                        'avg_code_quality': sum(r.code_quality_score for r in category_scores) / len(category_scores),
                        'avg_longcontext_utilization': sum(r.longcontext_utilization_score for r in category_scores) / len(category_scores)
                    }
            
            # Difficulty breakdown
            difficulty_results = {}
            for difficulty in DifficultyLevel:
                difficulty_name = difficulty.value
                difficulty_scores = [r for r in model_results if r.difficulty == difficulty_name]
                if difficulty_scores:
                    difficulty_results[difficulty_name] = {
                        'count': len(difficulty_scores),
                        'avg_total_score': sum(r.total_score for r in difficulty_scores) / len(difficulty_scores),
                        'avg_software_engineering': sum(r.software_engineering_score for r in difficulty_scores) / len(difficulty_scores),
                        'avg_functional_correctness': sum(r.functional_correctness_score for r in difficulty_scores) / len(difficulty_scores),
                        'avg_code_quality': sum(r.code_quality_score for r in difficulty_scores) / len(difficulty_scores),
                        'avg_longcontext_utilization': sum(r.longcontext_utilization_score for r in difficulty_scores) / len(difficulty_scores)
                    }
            
            summary = EvaluationSummary(
                model_name=model_name,
                total_scenarios=total_scenarios,
                completed_scenarios=completed_scenarios,
                failed_scenarios=failed_scenarios,
                
                avg_software_engineering_score=avg_software_engineering,
                avg_functional_correctness_score=avg_functional_correctness,
                avg_code_quality_score=avg_code_quality,
                avg_longcontext_utilization_score=avg_longcontext_utilization,
                avg_total_score=avg_total,
                
                avg_generation_time=avg_generation_time,
                total_evaluation_time=sum(r.generation_time for r in model_results),
                parsing_success_rate=parsing_success_rate,
                
                category_results=category_results,
                difficulty_results=difficulty_results
            )
            
            summaries[model_name] = summary
        
        return summaries
    
    def _get_letter_grade(self, score: float) -> str:
        """Convert numeric score to letter grade using config thresholds"""
        thresholds = self.config.phase4.score_thresholds
        
        if score >= thresholds["excellent"]["min"]:
            return "A (Excellent)"
        elif score >= thresholds["good"]["min"]:
            return "B (Good)" 
        elif score >= thresholds["fair"]["min"]:
            return "C (Fair)"
        else:
            return "F (Poor)"
    
    def _extract_software_engineering_metrics_summary(self, model_results: List[ModelEvaluationResult]) -> Dict[str, Any]:
        """Extract software engineering excellence metrics (8 metrics: ACS, DTA, CFRD, STS, RS, CS, IS, SES)"""
        from collections import defaultdict
        
        metrics_by_category = defaultdict(lambda: defaultdict(list))
        all_individual_scores = defaultdict(list)
        
        for result in model_results:
            category = result.task_category
            
            # Get software engineering metrics from both old and new structure (backward compatibility)
            se_details = result.detailed_results.get('software_engineering_details', {})
            individual_scores = se_details.get('individual_scores', {})
            
            # Initialize traditional_scores to avoid scope issues
            traditional_scores = {}
            
            # Also check old structure for backward compatibility
            if not individual_scores:
                # Get from traditional_agent_metrics_details (ACS, DTA, CFRD, ICU, MMR)
                traditional_details = result.detailed_results.get('traditional_agent_metrics_details', {})
                traditional_scores = traditional_details.get('individual_scores', {})
                
            # Get from advanced_metrics_details (STS, RS, CS, IS, SES)
            advanced_details = result.detailed_results.get('advanced_metrics_details', {})
            advanced_scores = advanced_details.get('individual_scores', {})
            
            # Combine into software engineering metrics (exclude ICU, MMR which go to LCU)
            se_metrics_from_traditional = {k: v for k, v in traditional_scores.items() 
                                         if k not in ['information_coverage_utilization', 'multi_session_memory_retention']}
            individual_scores = {**se_metrics_from_traditional, **advanced_scores}
            
            # Collect software engineering metric scores (8 metrics)
            for metric_name, score in individual_scores.items():
                all_individual_scores[metric_name].append(score)
                metrics_by_category[category][metric_name].append(score)
        
        # Calculate averages
        overall_averages = {}
        for metric_name, scores in all_individual_scores.items():
            if scores:
                overall_averages[metric_name] = {
                    'average': statistics.mean(scores),
                    'count': len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0
                }
        
        # Calculate category averages
        category_averages = {}
        for category, metrics in metrics_by_category.items():
            category_averages[category] = {}
            for metric_name, scores in metrics.items():
                if scores:
                    category_averages[category][metric_name] = {
                        'average': statistics.mean(scores),
                        'count': len(scores)
                    }
        
        return {
            'overall_averages': overall_averages,
            'category_breakdown': category_averages
        }
    
    def _extract_functional_correctness_metrics_summary(self, model_results: List[ModelEvaluationResult]) -> Dict[str, Any]:
        """Extract functional correctness metrics (4 metrics: Compilation, Unit Tests, Integration Tests, IDC)"""
        compilation_scores = []
        unit_test_scores = []
        integration_scores = []
        idc_scores = []
        
        for result in model_results:
            # Get functional correctness metrics from new or old structure (backward compatibility)
            fc_details = result.detailed_results.get('functional_correctness_details', {})
            
            # Extract functional sub-scores
            if 'overall_breakdown' in fc_details:
                breakdown = fc_details['overall_breakdown']
            else:
                # Fall back to old structure
                functional_details = result.detailed_results.get('functional_details', {})
                breakdown = functional_details.get('overall_breakdown', {})
            
            if breakdown:
                comp_score = breakdown.get('compilation_score', 0)
                unit_score = breakdown.get('unit_test_score', 0)
                integ_score = breakdown.get('integration_score', 0)
                
                compilation_scores.append(comp_score)
                unit_test_scores.append(unit_score)
                integration_scores.append(integ_score)
            
            # Get IDC from functional correctness breakdown (new structure) or fallback to old structure
            idc_score = None
            if breakdown and 'idc_score' in breakdown:
                idc_score = breakdown['idc_score']
            else:
                # Fall back to old structure - IDC might be in traditional_agent_metrics_details
                traditional_details = result.detailed_results.get('traditional_agent_metrics_details', {})
                individual_scores = traditional_details.get('individual_scores', {})
                idc_score = individual_scores.get('incremental_development_capability')
            
            if idc_score is not None:
                idc_scores.append(idc_score)
        
        # Calculate overall averages
        overall = {}
        if compilation_scores:
            overall['compilation'] = {
                'average': statistics.mean(compilation_scores),
                'count': len(compilation_scores)
            }
        if unit_test_scores:
            overall['unit_tests'] = {
                'average': statistics.mean(unit_test_scores),
                'count': len(unit_test_scores)
            }
        if integration_scores:
            overall['integration'] = {
                'average': statistics.mean(integration_scores),
                'count': len(integration_scores)
            }
        if idc_scores:
            overall['incremental_development_capability'] = {
                'average': statistics.mean(idc_scores),
                'count': len(idc_scores)
            }
        
        return {'overall_averages': overall}
    
    def _extract_code_quality_metrics_summary(self, model_results: List[ModelEvaluationResult]) -> Dict[str, Any]:
        """Extract code quality assessment metrics (3 metrics: Security, Quality Score, Issues Found)"""
        security_scores = []
        quality_scores = []
        issues_counts = []
        
        for result in model_results:
            # Get code quality metrics from new or old structure (backward compatibility)
            cq_details = result.detailed_results.get('code_quality_details', {})
            
            if cq_details:
                security_score = cq_details.get('security_analysis', {}).get('security_score', 0)
                overall_quality = cq_details.get('overall_quality_score', 0)
                issues_count = len(cq_details.get('issues_found', []))
            else:
                # Fall back to old structure
                quality_details = result.detailed_results.get('quality_details', {})
                security_score = quality_details.get('security_analysis', {}).get('security_score', 0)
                overall_quality = quality_details.get('overall_quality_score', 0)
                issues_count = len(quality_details.get('issues_found', []))
            
            security_scores.append(security_score)
            quality_scores.append(overall_quality)
            issues_counts.append(issues_count)
        
        # Calculate overall averages
        overall = {}
        if security_scores:
            overall['security'] = {
                'average': statistics.mean(security_scores),
                'count': len(security_scores)
            }
        if quality_scores:
            overall['overall_quality'] = {
                'average': statistics.mean(quality_scores),
                'count': len(quality_scores)
            }
        if issues_counts:
            overall['avg_issues_count'] = {
                'average': statistics.mean(issues_counts),
                'count': len(issues_counts)
            }
        
        return {'overall_averages': overall}
    
    def _extract_longcontext_utilization_metrics_summary(self, model_results: List[ModelEvaluationResult]) -> Dict[str, Any]:
        """Extract long-context utilization metrics (2 metrics: ICU, MMR)"""
        icu_scores = []
        mmr_scores = []
        
        for result in model_results:
            # Get long-context utilization metrics from new or old structure (backward compatibility)
            lcu_details = result.detailed_results.get('longcontext_utilization_details', {})
            individual_scores = lcu_details.get('individual_scores', {})
            
            # Fall back to old structure if needed
            if not individual_scores:
                traditional_details = result.detailed_results.get('traditional_agent_metrics_details', {})
                traditional_scores = traditional_details.get('individual_scores', {})
                # Only get ICU and MMR for long-context utilization
                icu_score = traditional_scores.get('information_coverage_utilization')
                mmr_score = traditional_scores.get('multi_session_memory_retention')
            else:
                icu_score = individual_scores.get('information_coverage_utilization')
                mmr_score = individual_scores.get('multi_session_memory_retention')
            
            if icu_score is not None:
                icu_scores.append(icu_score)
            if mmr_score is not None:
                mmr_scores.append(mmr_score)
        
        # Calculate overall averages
        overall = {}
        if icu_scores:
            overall['information_coverage_utilization'] = {
                'average': statistics.mean(icu_scores),
                'count': len(icu_scores)
            }
        if mmr_scores:
            overall['multi_session_memory_retention'] = {
                'average': statistics.mean(mmr_scores),
                'count': len(mmr_scores)
            }
        
        return {'overall_averages': overall}
    
    def save_markdown_summary(self, summaries: Dict[str, EvaluationSummary], 
                             results: List[ModelEvaluationResult], output_file: Path):
        """Save a clear and well-organized markdown summary of evaluation results"""
        
        markdown_content = []
        
        # Header
        markdown_content.append("# 📊 LoCoBench Results Summary")
        markdown_content.append("")
        markdown_content.append(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append(f"**Framework Version:** LoCoBench v1.0")
        markdown_content.append(f"**Benchmark:** Multi-language Software Development Tasks")
        markdown_content.append("")
        
        # Overall Model Performance Table
        markdown_content.append("## 🏆 Model Performance Comparison")
        markdown_content.append("")
        
        # Create the main performance table
        table_headers = ["Model", "Total Score", "Grade", "Software Engineering", "Functional Correctness", "Code Quality", "Long-Context Util", "Success Rate"]
        markdown_content.append("| " + " | ".join(table_headers) + " |")
        markdown_content.append("| " + " | ".join(["---"] * len(table_headers)) + " |")
        
        # Sort models by total score (descending)
        sorted_models = sorted(summaries.items(), key=lambda x: x[1].avg_total_score, reverse=True)
        
        for i, (model_name, summary) in enumerate(sorted_models):
            # Add medal emoji for top performers
            model_display = model_name
            if i == 0:
                model_display = f"🥇 {model_name}"
            elif i == 1:
                model_display = f"🥈 {model_name}"
            elif i == 2:
                model_display = f"🥉 {model_name}"
            
            grade = self._get_letter_grade(summary.avg_total_score)
            success_rate = f"{(summary.completed_scenarios / summary.total_scenarios):.1%}" if summary.total_scenarios > 0 else "0.0%"
            
            row_data = [
                model_display,
                f"{summary.avg_total_score:.3f}",
                grade,
                f"{summary.avg_software_engineering_score:.3f}",
                f"{summary.avg_functional_correctness_score:.3f}",
                f"{summary.avg_code_quality_score:.3f}",
                f"{summary.avg_longcontext_utilization_score:.3f}",
                success_rate
            ]
            
            markdown_content.append("| " + " | ".join(row_data) + " |")
        
        markdown_content.append("")
        
        # DETAILED MODEL ANALYSIS
        for model_name, summary in sorted_models:
            # Get detailed results for this model
            model_results = [r for r in results if r.model_name == model_name]
            
            if not model_results:
                continue
                
            markdown_content.append(f"## 🤖 {model_name.upper()} - Detailed Analysis")
            markdown_content.append("")
            
            # EXECUTIVE SUMMARY for this model
            markdown_content.append("### 📈 Performance Overview")
            markdown_content.append("")
            markdown_content.append(f"- **🎯 Total Score:** {summary.avg_total_score:.3f} ({self._get_letter_grade(summary.avg_total_score)})")
            markdown_content.append(f"- **🏗️ Software Engineering:** {summary.avg_software_engineering_score:.3f} (Weight: 40%)")
            markdown_content.append(f"- **⚙️ Functional Correctness:** {summary.avg_functional_correctness_score:.3f} (Weight: 30%)")
            markdown_content.append(f"- **🔍 Code Quality:** {summary.avg_code_quality_score:.3f} (Weight: 20%)")
            markdown_content.append(f"- **🧠 Long-Context Utilization:** {summary.avg_longcontext_utilization_score:.3f} (Weight: 10%)")
            markdown_content.append(f"- **✅ Success Rate:** {(summary.completed_scenarios / summary.total_scenarios):.1%}")
            markdown_content.append("")
            
            # Extract detailed metrics using the new 4-dimension structure
            software_engineering_metrics = self._extract_software_engineering_metrics_summary(model_results)
            functional_correctness_metrics = self._extract_functional_correctness_metrics_summary(model_results)
            code_quality_metrics = self._extract_code_quality_metrics_summary(model_results)
            longcontext_utilization_metrics = self._extract_longcontext_utilization_metrics_summary(model_results)
            
            # Get all detailed metrics data
            software_engineering_overall = software_engineering_metrics['overall_averages']
            functional_correctness_overall = functional_correctness_metrics['overall_averages']
            code_quality_overall = code_quality_metrics['overall_averages']
            longcontext_utilization_overall = longcontext_utilization_metrics['overall_averages']
            
            # 1. SOFTWARE ENGINEERING EXCELLENCE (8 metrics)
            markdown_content.append("### 🏗️ Software Engineering Excellence (8 metrics)")
            markdown_content.append("")
            markdown_content.append("*Advanced software engineering practices and architectural understanding*")
            markdown_content.append("")
            markdown_content.append("| Metric | Score | Description |")
            markdown_content.append("|--------|-------|-------------|")
            
            # Define Software Engineering metrics in order
            se_metrics = [
                ('architectural_coherence_score', 'ACS', 'System design consistency and architectural principles'),
                ('dependency_traversal_accuracy', 'DTA', 'Cross-module dependency navigation ability'),
                ('cross_file_reasoning_depth', 'CFRD', 'Multi-file relationship understanding'),
                ('system_thinking_score', 'STS', 'Scalability and maintainability considerations'),
                ('robustness_score', 'RS', 'Error handling and security practices'),
                ('comprehensiveness_score', 'CS', 'Documentation and API completeness'),
                ('innovation_score', 'IS', 'Modern patterns and algorithmic efficiency'),
                ('solution_elegance_score', 'SES', 'Code clarity and abstraction appropriateness')
            ]
            
            for metric_key, short_name, description in se_metrics:
                if metric_key in software_engineering_overall:
                    stats = software_engineering_overall[metric_key]
                    markdown_content.append(f"| **{short_name}** | {stats['average']:.3f} | {description} |")
            
            markdown_content.append("")
            
            # 2. FUNCTIONAL CORRECTNESS (4 metrics)
            markdown_content.append("### ⚙️ Functional Correctness (4 metrics)")
            markdown_content.append("")
            markdown_content.append("*Code compilation, testing, and incremental development*")
            markdown_content.append("")
            markdown_content.append("| Metric | Score | Description |")
            markdown_content.append("|--------|-------|-------------|")
            
            # Functional metrics
            functional_metrics_list = [
                ('compilation', 'Code Compilation', 'Successful compilation across languages'),
                ('unit_tests', 'Unit Tests', 'Unit test execution and passing rate'),
                ('integration', 'Integration Tests', 'End-to-end functionality validation'),
                ('incremental_development_capability', 'IDC', 'Building effectively on previous work')
            ]
            
            for metric_key, display_name, description in functional_metrics_list:
                if metric_key in functional_correctness_overall:
                    stats = functional_correctness_overall[metric_key]
                    markdown_content.append(f"| **{display_name}** | {stats['average']:.3f} | {description} |")
            
            markdown_content.append("")
            
            # 3. CODE QUALITY ASSESSMENT (3 metrics)
            markdown_content.append("### 🔍 Code Quality Assessment (3 metrics)")
            markdown_content.append("")
            markdown_content.append("*Static analysis, security, and maintainability*")
            markdown_content.append("")
            markdown_content.append("| Metric | Score | Description |")
            markdown_content.append("|--------|-------|-------------|")
            
            quality_metrics_list = [
                ('security', 'Security Analysis', 'Vulnerability detection and security practices'),
                ('overall_quality', 'Code Quality', 'Overall maintainability and readability'),
                ('avg_issues_count', 'Issues Found', 'Average code issues detected (lower is better)')
            ]
            
            for metric_key, display_name, description in quality_metrics_list:
                if metric_key in code_quality_overall:
                    stats = code_quality_overall[metric_key]
                    markdown_content.append(f"| **{display_name}** | {stats['average']:.3f} | {description} |")
            
            markdown_content.append("")
            
            # 4. LONG-CONTEXT UTILIZATION (2 metrics)
            markdown_content.append("### 🧠 Long-Context Utilization (2 metrics)")
            markdown_content.append("")
            markdown_content.append("*Context usage efficiency and memory retention*")
            markdown_content.append("")
            markdown_content.append("| Metric | Score | Description |")
            markdown_content.append("|--------|-------|-------------|")
            
            longcontext_util_metrics = [
                ('information_coverage_utilization', 'ICU', 'Effective usage of provided context information'),
                ('multi_session_memory_retention', 'MMR', 'Context persistence across development sessions')
            ]
            
            for metric_key, short_name, description in longcontext_util_metrics:
                if metric_key in longcontext_utilization_overall:
                    stats = longcontext_utilization_overall[metric_key]
                    markdown_content.append(f"| **{short_name}** | {stats['average']:.3f} | {description} |")
            
            markdown_content.append("")
            
            # PERFORMANCE BY TASK CATEGORY
            markdown_content.append("### 📊 Performance by Task Category")
            markdown_content.append("")
            
            # Best and worst performing categories
            if summary.category_results:
                best_category = max(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'])
                worst_category = min(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'])
                
                markdown_content.append(f"- **🏆 Strongest Category:** {best_category[0].replace('_', ' ').title()} ({best_category[1]['avg_total_score']:.3f})")
                markdown_content.append(f"- **📈 Improvement Area:** {worst_category[0].replace('_', ' ').title()} ({worst_category[1]['avg_total_score']:.3f})")
            markdown_content.append("")
            
            # Category breakdown table
            markdown_content.append("| Category | Total Score | Software Engineering | Scenarios |")
            markdown_content.append("|----------|-------------|---------------------|-----------|")
            
            for category, stats in sorted(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'], reverse=True):
                category_name = category.replace('_', ' ').title()
                markdown_content.append(f"| {category_name} | {stats['avg_total_score']:.3f} | {stats['avg_software_engineering']:.3f} | {stats['count']} |")
            
            markdown_content.append("")
            markdown_content.append("---")
            markdown_content.append("")
        
        # CROSS-MODEL INSIGHTS (only if multiple models)
        if len(summaries) > 1:
            markdown_content.append("## 🔄 Multi-Model Comparison")
            markdown_content.append("")
            
            # Get all unique categories
            all_categories = set()
            for summary in summaries.values():
                all_categories.update(summary.category_results.keys())
            
            for category in sorted(all_categories):
                category_title = category.replace('_', ' ').title()
                markdown_content.append(f"### {category_title}")
                markdown_content.append("")
                
                # Category table headers
                cat_headers = ["Model", "Count", "Total Score", "Software Engineering", "Performance"]
                markdown_content.append("| " + " | ".join(cat_headers) + " |")
                markdown_content.append("| " + " | ".join(["---"] * len(cat_headers)) + " |")
                
                for model_name, summary in sorted_models:
                    if category in summary.category_results:
                        cat_result = summary.category_results[category]
                        cat_row = [
                            model_name,
                            str(cat_result['count']),
                            f"{cat_result['avg_total_score']:.3f}",
                            f"{cat_result['avg_software_engineering']:.3f}",
                            "✅" if cat_result['avg_software_engineering'] > 0.3 else "📈"
                        ]
                        markdown_content.append("| " + " | ".join(cat_row) + " |")
                
                markdown_content.append("")
        
        # SUMMARY INSIGHTS
        markdown_content.append("## 💡 Summary & Insights")
        markdown_content.append("")
        
        if len(summaries) > 0:
            # Overall statistics
            total_evaluations = sum(s.completed_scenarios for s in summaries.values())
            avg_success_rate = sum((s.completed_scenarios / s.total_scenarios) if s.total_scenarios > 0 else 0 for s in summaries.values()) / len(summaries)
            
            markdown_content.append("### 📊 Evaluation Statistics")
            markdown_content.append("")
            markdown_content.append(f"- **📈 Total Evaluations:** {total_evaluations:,} scenarios")
            markdown_content.append(f"- **✅ Success Rate:** {avg_success_rate:.1%}")
            markdown_content.append(f"- **🎯 Coverage:** 8 task categories across multiple difficulty levels")
            markdown_content.append("")
            
            if len(summaries) == 1:
                # Single model insights
                model_name, summary = list(summaries.items())[0]
                
                markdown_content.append("### 🎯 Key Findings")
                markdown_content.append("")
                markdown_content.append(f"- **Overall Performance:** {summary.avg_total_score:.3f} ({self._get_letter_grade(summary.avg_total_score)})")
                
                # Strengths and weaknesses
                dimension_scores = [
                    ("Software Engineering", summary.avg_software_engineering_score),
                    ("Functional Correctness", summary.avg_functional_correctness_score), 
                    ("Code Quality", summary.avg_code_quality_score),
                    ("Long-Context Utilization", summary.avg_longcontext_utilization_score)
                ]
                
                best_dimension = max(dimension_scores, key=lambda x: x[1])
                worst_dimension = min(dimension_scores, key=lambda x: x[1])
                
                markdown_content.append(f"- **Strongest Dimension:** {best_dimension[0]} ({best_dimension[1]:.3f})")
                markdown_content.append(f"- **Improvement Area:** {worst_dimension[0]} ({worst_dimension[1]:.3f})")
                
                # Category performance
                if summary.category_results:
                    best_category = max(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'])
                    worst_category = min(summary.category_results.items(), key=lambda x: x[1]['avg_total_score'])
                    
                    markdown_content.append(f"- **Best Category:** {best_category[0].replace('_', ' ').title()} ({best_category[1]['avg_total_score']:.3f})")
                    markdown_content.append(f"- **Challenging Category:** {worst_category[0].replace('_', ' ').title()} ({worst_category[1]['avg_total_score']:.3f})")
            else:
                # Multi-model insights
                best_model = sorted_models[0][0]
                best_score = sorted_models[0][1].avg_total_score
                
                markdown_content.append("### 🏆 Model Ranking")
                markdown_content.append("")
                markdown_content.append(f"- **🥇 Top Performer:** {best_model} ({best_score:.3f})")
                
                if len(sorted_models) > 1:
                    score_gap = sorted_models[0][1].avg_total_score - sorted_models[1][1].avg_total_score
                    markdown_content.append(f"- **Performance Gap:** {score_gap:.3f} points between 1st and 2nd place")
        
        markdown_content.append("")
        markdown_content.append("## 📖 Evaluation Framework")
        markdown_content.append("")
        markdown_content.append("LoCoBench uses **17 metrics across 4 dimensions**:")
        markdown_content.append("")
        markdown_content.append("1. **Software Engineering Excellence (40%)** - 8 metrics")
        markdown_content.append("   - Architectural Coherence Score (ACS)")
        markdown_content.append("   - Dependency Traversal Accuracy (DTA)")
        markdown_content.append("   - Cross-File Reasoning Depth (CFRD)")
        markdown_content.append("   - System Thinking Score (STS)")
        markdown_content.append("   - Robustness Score (RS)")
        markdown_content.append("   - Comprehensiveness Score (CS)")
        markdown_content.append("   - Innovation Score (IS)")
        markdown_content.append("   - Solution Elegance Score (SES)")
        markdown_content.append("")
        markdown_content.append("2. **Functional Correctness (30%)** - 4 metrics")
        markdown_content.append("   - Code Compilation")
        markdown_content.append("   - Unit Tests")
        markdown_content.append("   - Integration Tests")
        markdown_content.append("   - Incremental Development Capability (IDC)")
        markdown_content.append("")
        markdown_content.append("3. **Code Quality Assessment (20%)** - 3 metrics")
        markdown_content.append("   - Security Analysis")
        markdown_content.append("   - Code Quality")
        markdown_content.append("   - Issues Found")
        markdown_content.append("")
        markdown_content.append("4. **Long-Context Utilization (10%)** - 2 metrics")
        markdown_content.append("   - Information Coverage Utilization (ICU)")
        markdown_content.append("   - Multi-Session Memory Retention (MMR)")
        markdown_content.append("")
        
        # Write to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_content))
        
        print(f"✅ Markdown summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate markdown summary from evaluation results')
    parser.add_argument('--config-path', '-c', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--results-file', '-r', type=str, 
                       default=None,
                       help='Path to evaluation results JSONL file (default: auto-detect from config)')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                       help='Output markdown file path (default: evaluation_results/summary_TIMESTAMP.md)')
    
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config_path)
    
    # Determine results file path
    if args.results_file:
        results_file = Path(args.results_file)
        if not results_file.is_absolute():
            results_file = Path(__file__).parent.parent / results_file
    else:
        # Auto-detect: try common locations
        base_dir = Path(__file__).parent.parent
        intermediate_dir = base_dir / "intermediate_results"
        
        # First, try to find model-specific files and combine them
        results_files = []
        if intermediate_dir.exists():
            model_specific_files = list(intermediate_dir.glob("evaluation_incremental_results_*.json"))
            if model_specific_files:
                print(f"📂 Found {len(model_specific_files)} model-specific result files")
                results_files = model_specific_files
            else:
                # Try default location
                default_file = intermediate_dir / "evaluation_incremental_results.json"
                if default_file.exists():
                    results_files = [default_file]
        else:
            default_file = intermediate_dir / "evaluation_incremental_results.json"
            if default_file.exists():
                results_files = [default_file]
        
        if not results_files:
            print(f"❌ No results files found in: {intermediate_dir}")
            print(f"   Use --results-file to specify the correct path")
            sys.exit(1)
        
        results_file = results_files  # Pass as list to load_results_from_jsonl
    
    # Determine output file path
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).parent.parent / "evaluation_results" / f"summary_{timestamp}.md"
    
    if isinstance(results_file, list):
        print(f"📂 Loading results from {len(results_file)} file(s):")
        for f in results_file:
            print(f"   • {f.name}")
    else:
        print(f"📂 Loading results from: {results_file}")
    print(f"📝 Output will be saved to: {output_file}")
    print()
    
    # Create generator
    generator = MarkdownSummaryGenerator(config)
    
    # Load results
    all_results = generator.load_results_from_jsonl(results_file)
    
    if not all_results:
        print("❌ No evaluation results found in the file.")
        sys.exit(1)
    
    # Group results by model
    results_by_model = {}
    for result in all_results:
        if result.model_name not in results_by_model:
            results_by_model[result.model_name] = []
        results_by_model[result.model_name].append(result)
    
    print(f"🤖 Found results for {len(results_by_model)} model(s): {list(results_by_model.keys())}")
    print()
    
    # Generate summaries
    print("📊 Generating evaluation summaries...")
    summaries = generator.generate_evaluation_summary(results_by_model)
    
    if not summaries:
        print("❌ No summaries generated")
        sys.exit(1)
    
    # Save markdown summary
    print("💾 Saving markdown summary...")
    generator.save_markdown_summary(summaries, all_results, output_file)
    
    print()
    print("✅ Markdown summary generation complete!")


if __name__ == '__main__':
    main()
