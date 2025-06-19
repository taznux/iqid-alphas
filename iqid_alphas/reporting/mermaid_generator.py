"""
Mermaid Diagram Generation

Creates dynamic Mermaid flowcharts for pipeline visualization.
Supports color-coding based on step success/failure states.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class StepStatus(Enum):
    """Status of a pipeline step for visualization."""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    RUNNING = "running"
    PENDING = "pending"


@dataclass
class PipelineStep:
    """Represents a single pipeline step for diagram generation."""
    id: str
    name: str
    status: StepStatus
    duration: Optional[float] = None
    details: Optional[str] = None


class MermaidGenerator:
    """Generates Mermaid diagrams for pipeline visualization."""
    
    def __init__(self):
        """Initialize the Mermaid generator with styling."""
        self.colors = {
            StepStatus.SUCCESS: "#d4edda,stroke:#c3e6cb",
            StepStatus.FAILURE: "#f8d7da,stroke:#f5c6cb",
            StepStatus.WARNING: "#fff3cd,stroke:#ffeaa7",
            StepStatus.RUNNING: "#cce5ff,stroke:#99ccff",
            StepStatus.PENDING: "#f8f9fa,stroke:#dee2e6"
        }
    
    def generate_pipeline_flowchart(self, steps: List[PipelineStep], 
                                  title: str = "Pipeline Flow") -> str:
        """
        Generate a Mermaid flowchart for pipeline steps.
        
        Args:
            steps: List of pipeline steps
            title: Title for the flowchart
            
        Returns:
            Mermaid diagram as string
        """
        if not steps:
            return self._generate_empty_diagram(title)
        
        # Build the flowchart
        diagram_lines = [
            f"```mermaid",
            f"graph TD"
        ]
        
        # Add nodes and connections
        for i, step in enumerate(steps):
            node_id = self._sanitize_id(step.id)
            node_label = self._format_node_label(step)
            
            if i == 0:
                # First node
                diagram_lines.append(f"    {node_id}[{node_label}]")
            else:
                # Connected nodes
                prev_node_id = self._sanitize_id(steps[i-1].id)
                diagram_lines.append(f"    {prev_node_id} --> {node_id}[{node_label}]")
        
        # Add styling
        diagram_lines.append("")
        for step in steps:
            node_id = self._sanitize_id(step.id)
            color = self.colors.get(step.status, self.colors[StepStatus.PENDING])
            diagram_lines.append(f"    style {node_id} fill:{color}")
        
        diagram_lines.append("```")
        return "\n".join(diagram_lines)
    
    def generate_success_pie_chart(self, success_counts: Dict[str, int]) -> str:
        """
        Generate a pie chart showing success/failure rates.
        
        Args:
            success_counts: Dictionary with success/failure counts
            
        Returns:
            Mermaid pie chart as string
        """
        total = sum(success_counts.values())
        if total == 0:
            return "```mermaid\npie title No Data Available\n    \"No samples\" : 1\n```"
        
        diagram_lines = [
            "```mermaid",
            f"pie title Success Rate ({total} samples)"
        ]
        
        for label, count in success_counts.items():
            diagram_lines.append(f'    "{label}" : {count}')
        
        diagram_lines.append("```")
        return "\n".join(diagram_lines)
    
    def generate_timeline_diagram(self, events: List[Dict[str, Any]]) -> str:
        """
        Generate a timeline diagram for batch processing.
        
        Args:
            events: List of timeline events with timestamps
            
        Returns:
            Mermaid timeline diagram as string
        """
        if not events:
            return "```mermaid\ntimeline\n    title Empty Timeline\n```"
        
        diagram_lines = [
            "```mermaid",
            "timeline",
            "    title Processing Timeline"
        ]
        
        for event in events:
            timestamp = event.get('timestamp', 'Unknown')
            description = event.get('description', 'Event')
            diagram_lines.append(f'    {timestamp} : {description}')
        
        diagram_lines.append("```")
        return "\n".join(diagram_lines)
    
    def _sanitize_id(self, node_id: str) -> str:
        """Sanitize node ID for Mermaid compatibility."""
        # Replace spaces and special characters with underscores
        sanitized = "".join(c if c.isalnum() else "_" for c in node_id)
        # Ensure it starts with a letter
        if not sanitized[0].isalpha():
            sanitized = "N" + sanitized
        return sanitized
    
    def _format_node_label(self, step: PipelineStep) -> str:
        """Format node label with status indicators."""
        status_icons = {
            StepStatus.SUCCESS: "✅",
            StepStatus.FAILURE: "❌",
            StepStatus.WARNING: "⚠️",
            StepStatus.RUNNING: "🔄",
            StepStatus.PENDING: "⏸️"
        }
        
        icon = status_icons.get(step.status, "")
        label = f"{step.name}"
        
        if step.duration:
            label += f"<br/>{step.duration:.1f}s"
        
        return f"{icon} {label}"
    
    def _generate_empty_diagram(self, title: str) -> str:
        """Generate an empty diagram placeholder."""
        return f"""```mermaid
graph TD
    A[{title}]
    A --> B[No steps to display]
    
    style A fill:#f8f9fa,stroke:#dee2e6
    style B fill:#f8f9fa,stroke:#dee2e6
```"""


class PipelineFlowChart:
    """Specialized flowchart generator for pipeline stages."""
    
    def __init__(self):
        self.generator = MermaidGenerator()
    
    def create_segmentation_flow(self, results: Dict[str, Any]) -> str:
        """Create flowchart for segmentation pipeline."""
        steps = [
            PipelineStep("load", "Load Raw Image", 
                        StepStatus.SUCCESS if results.get('load_success') else StepStatus.FAILURE),
            PipelineStep("segment", "Segment Tissues",
                        StepStatus.SUCCESS if results.get('segment_success') else StepStatus.FAILURE),
            PipelineStep("validate", "Validate Segmentation",
                        StepStatus.SUCCESS if results.get('validate_success') else StepStatus.FAILURE)
        ]
        
        return self.generator.generate_pipeline_flowchart(steps, "Segmentation Pipeline")
    
    def create_alignment_flow(self, results: Dict[str, Any]) -> str:
        """Create flowchart for alignment pipeline."""
        steps = [
            PipelineStep("load", "Load Segmented Images",
                        StepStatus.SUCCESS if results.get('load_success') else StepStatus.FAILURE),
            PipelineStep("align", "Align Image Stack",
                        StepStatus.SUCCESS if results.get('align_success') else StepStatus.FAILURE),
            PipelineStep("validate", "Validate Alignment",
                        StepStatus.SUCCESS if results.get('validate_success') else StepStatus.FAILURE)
        ]
        
        return self.generator.generate_pipeline_flowchart(steps, "Alignment Pipeline")
    
    def create_coregistration_flow(self, results: Dict[str, Any]) -> str:
        """Create flowchart for coregistration pipeline."""
        steps = [
            PipelineStep("pair", "Pair Modalities",
                        StepStatus.SUCCESS if results.get('pair_success') else StepStatus.FAILURE),
            PipelineStep("register", "Register Images",
                        StepStatus.SUCCESS if results.get('register_success') else StepStatus.FAILURE),
            PipelineStep("assess", "Assess Quality",
                        StepStatus.SUCCESS if results.get('assess_success') else StepStatus.FAILURE)
        ]
        
        return self.generator.generate_pipeline_flowchart(steps, "Coregistration Pipeline")
