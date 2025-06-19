#!/usr/bin/env python3
"""
Coregistration Pipeline Module

Main pipeline class for iQID + H&E multi-modal coregistration.
Modularized for maintainability and follows <500 line guideline.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from ..base import BasePipeline
from .config import CoregistrationPipelineConfig
from .processor import CoregistrationProcessor
from .utils import CoregistrationDataLoader, CoregistrationMetricsCalculator
from ...core.batch_processing import BatchProcessor


class CoregistrationPipeline(BasePipeline):
    """
    Co-registration Pipeline for DataPush1 Dataset
    
    Develops and evaluates multi-modal co-registration between aligned iQID 
    and H&E images using proxy quality metrics.
    """
    
    def __init__(self, data_path: Union[str, Path], output_dir: Union[str, Path],
                 config: Optional[CoregistrationPipelineConfig] = None,
                 registration_method: str = "mutual_information",
                 pairing_strategy: str = "automatic",
                 config_path: Optional[Path] = None):
        """
        Initialize co-registration pipeline
        
        Args:
            data_path: Path to DataPush1 dataset
            output_dir: Output directory for results
            config: Pipeline configuration (optional)
            registration_method: Registration method (mutual_information, feature_based, intensity_based)
            pairing_strategy: Sample pairing strategy (automatic, manual, fuzzy_matching)
            config_path: Optional configuration file path
        """
        # Load configuration if not provided
        if config is None:
            config = CoregistrationPipelineConfig.from_config_file(config_path)
            
        # Override config with provided parameters
        if registration_method:
            config.registration_method = registration_method
        if pairing_strategy:
            config.pairing_strategy = pairing_strategy
        
        super().__init__(data_path, output_dir, config)
        
        # Initialize components
        self.processor = CoregistrationProcessor(config)
        self.data_loader = CoregistrationDataLoader(config)
        self.metrics_calculator = CoregistrationMetricsCalculator(config)
        self.batch_processor = BatchProcessor(str(data_path), str(output_dir))
        
        self.logger.info(f"Initialized CoregistrationPipeline")
        self.logger.info(f"Registration method: {config.registration_method}")
        self.logger.info(f"Pairing strategy: {config.pairing_strategy}")
    
    def discover_samples(self) -> List[Dict[str, Any]]:
        """
        Discover and pair iQID and H&E samples for coregistration.
        
        Returns:
            List of sample pairs with metadata
        """
        self.logger.info("Discovering and pairing samples...")
        
        data_path = Path(self.data_path)
        sample_pairs = self.data_loader.discover_samples(data_path)
        
        self.logger.info(f"Found {len(sample_pairs)} sample pairs for coregistration")
        return sample_pairs
    
    def process_sample(self, sample: Dict[str, Any]):
        """
        Process a single coregistration sample pair.
        
        Args:
            sample: Sample dictionary with paired iQID and H&E directories
            
        Returns:
            ValidationResult with coregistration outcomes
        """
        return self.processor.process_sample(sample, self.logger)
    
    def run_evaluation(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete co-registration evaluation pipeline.
        
        This method provides compatibility with the existing CLI interface
        while using the new modular architecture.
        
        Args:
            max_samples: Maximum number of sample pairs to process
            
        Returns:
            Dictionary containing evaluation results and summary statistics
        """
        self.logger.info("Starting co-registration evaluation pipeline")
        
        try:
            # Run the main pipeline
            result = self.run_pipeline(max_samples)
            
            # Generate additional coregistration-specific summary
            summary = self.metrics_calculator.calculate_summary_statistics(
                [self._convert_validation_to_coreg_result(vr) for vr in result.validation_results]
            )
            
            # Add pipeline result information
            summary.update({
                "pipeline_name": "CoregistrationPipeline",
                "total_samples": result.total_samples,
                "successful_samples": result.successful_samples,
                "failed_samples": result.failed_samples,
                "success_rate": result.success_rate,
                "duration": result.duration,
                "output_dir": str(self.output_dir)
            })
            
            self.logger.info("Co-registration evaluation pipeline completed")
            return summary
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {'error': str(e), 'status': 'failed'}
    
    def _convert_validation_to_coreg_result(self, validation_result):
        """Convert ValidationResult back to CoregistrationResult for metrics calculation."""
        from .utils import CoregistrationResult
        
        # This is a helper method to maintain compatibility with existing metrics calculation
        metrics = validation_result.metrics
        
        return CoregistrationResult(
            pair_id=getattr(validation_result, 'sample_id', 'unknown'),
            iqid_dir=metrics.get('iqid_dir', ''),
            he_dir=metrics.get('he_dir', ''),
            success=validation_result.success,
            registration_method=metrics.get('registration_method', self.config.registration_method),
            pairing_confidence=metrics.get('pairing_confidence', 0.0),
            processing_time=validation_result.processing_time,
            iqid_slice_count=metrics.get('iqid_slice_count', 0),
            he_slice_count=metrics.get('he_slice_count', 0),
            mutual_information=metrics.get('mutual_information', 0.0),
            structural_similarity=metrics.get('structural_similarity', 0.0),
            registration_confidence=metrics.get('registration_confidence', 0.0),
            feature_matches=metrics.get('feature_matches', 0),
            transform_parameters=metrics.get('transform_parameters', {}),
            error_message=getattr(validation_result, 'error_message', None)
        )
    
    def run(self) -> Dict[str, Any]:
        """
        Run the coregistration pipeline.
        Returns a result dictionary with keys:
            - total_pairs
            - successful_pairings
            - failed_pairs
            - success_rate
            - duration
            - output_dir
            - metrics (optional)
        """
        self.logger.info("Running the coregistration pipeline...")
        
        # Implement the standardized pipeline logic here
        # This should include discovery, processing, and evaluation of samples
        
        # For now, return a dummy result
        return {
            'total_pairs': 0,
            'successful_pairings': 0,
            'failed_pairs': 0,
            'success_rate': 0.0,
            'duration': 0.0,
            'output_dir': str(self.output_dir),
            'metrics': {}
        }
