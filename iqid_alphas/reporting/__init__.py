"""
Reporting Package

Comprehensive visual reporting system for IQID-Alphas pipelines.
Generates rich Markdown reports with Mermaid diagrams and embedded visualizations.

Components:
- report_generator: Main report generation orchestration
- mermaid_generator: Dynamic Mermaid diagram creation
- templates: Report templates and formatting
- styles: Visualization styling and color schemes

Usage:
    from iqid_alphas.reporting import ReportGenerator
    
    generator = ReportGenerator(output_dir="reports")
    report = generator.generate_sample_report(sample_results, visualizations)
"""

from .report_generator import ReportGenerator, SampleReport, BatchReport
from .mermaid_generator import MermaidGenerator, PipelineFlowChart

__all__ = [
    'ReportGenerator',
    'SampleReport', 
    'BatchReport',
    'MermaidGenerator',
    'PipelineFlowChart'
]
