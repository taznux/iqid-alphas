"""
Unified Aligner API

Provides a common interface for all alignment functionality including
simple alignment, enhanced alignment, and stack processing.
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np

from .simple import SimpleAligner
from .enhanced import EnhancedAligner
from .stack import StackAligner
from ..types import AlignmentMethod, ReferenceStrategy, AlignmentResult, StackAlignmentResult


class UnifiedAligner:
    """
    Unified interface for all alignment functionality.
    
    Provides access to simple alignment, enhanced alignment, and stack processing
    through a single, consistent API.
    """
    
    def __init__(self, 
                 default_method: AlignmentMethod = AlignmentMethod.PHASE_CORRELATION,
                 reference_strategy: ReferenceStrategy = ReferenceStrategy.PREVIOUS_IMAGE):
        """
        Initialize the unified aligner.
        
        Parameters
        ----------
        default_method : AlignmentMethod
            Default alignment method for single image alignment
        reference_strategy : ReferenceStrategy
            Default reference strategy for stack alignment
        """
        self.default_method = default_method
        self.reference_strategy = reference_strategy
        
        # Initialize component aligners
        self.simple_aligner = SimpleAligner()
        self.enhanced_aligner = EnhancedAligner()
        self.stack_aligner = StackAligner(reference_strategy=reference_strategy)
    
    def align_images(self, 
                    moving_image: np.ndarray,
                    fixed_image: np.ndarray,
                    method: Optional[AlignmentMethod] = None,
                    use_enhanced: bool = True) -> AlignmentResult:
        """
        Align two images using either simple or enhanced alignment.
        
        Parameters
        ----------
        moving_image : np.ndarray
            Image to be aligned
        fixed_image : np.ndarray
            Reference image
        method : AlignmentMethod, optional
            Alignment method to use
        use_enhanced : bool
            Whether to use enhanced aligner (default) or simple aligner
            
        Returns
        -------
        AlignmentResult
            Alignment result with transformation and quality metrics
        """
        method = method or self.default_method
        
        if use_enhanced:
            return self.enhanced_aligner.align_slice(moving_image, fixed_image, method)
        else:
            # Use simple aligner
            aligned_image, metadata = self.simple_aligner.align_images(fixed_image, moving_image)
            
            # Convert to AlignmentResult format
            return AlignmentResult(
                transform_matrix=metadata.get('transform_matrix', np.eye(3)),
                aligned_image=aligned_image,
                confidence=metadata.get('confidence', 0.5),
                method=method,
                metadata=metadata
            )
    
    def align_stack(self,
                   image_stack: np.ndarray,
                   reference_strategy: Optional[ReferenceStrategy] = None) -> StackAlignmentResult:
        """
        Align an image stack.
        
        Parameters
        ----------
        image_stack : np.ndarray
            Stack of images to align (shape: [n_images, height, width])
        reference_strategy : ReferenceStrategy, optional
            Strategy for selecting reference images
            
        Returns
        -------
        StackAlignmentResult
            Complete stack alignment result
        """
        return self.stack_aligner.align_stack(image_stack, reference_strategy)
    
    def align_with_padding(self,
                          images: List[np.ndarray],
                          padding_factor: float = 2.0) -> tuple:
        """
        Align images with padding using simple aligner.
        
        Parameters
        ----------
        images : List[np.ndarray]
            List of images to align
        padding_factor : float
            Padding factor for alignment
            
        Returns
        -------
        tuple
            (aligned_images, transforms)
        """
        return self.simple_aligner.align_stack_with_padding(images, padding_factor)
    
    def align_simple_stack(self, images: List[np.ndarray]) -> tuple:
        """
        Simple stack alignment using basic aligner.
        
        Parameters
        ----------
        images : List[np.ndarray]
            List of images to align
            
        Returns
        -------
        tuple
            (aligned_images, transforms)
        """
        return self.simple_aligner.align_stack_simple(images)
    
    def get_alignment_quality(self, 
                             fixed: np.ndarray, 
                             aligned: np.ndarray) -> Dict[str, float]:
        """
        Calculate alignment quality metrics.
        
        Parameters
        ----------
        fixed : np.ndarray
            Reference image
        aligned : np.ndarray
            Aligned image
            
        Returns
        -------
        Dict[str, float]
            Quality metrics
        """
        return self.simple_aligner.calculate_alignment_quality(fixed, aligned)
    
    def set_default_method(self, method: AlignmentMethod):
        """Set the default alignment method."""
        self.default_method = method
    
    def set_reference_strategy(self, strategy: ReferenceStrategy):
        """Set the default reference strategy for stack alignment."""
        self.reference_strategy = strategy
        self.stack_aligner.reference_strategy = strategy


# Convenience functions for backward compatibility
def align_images(moving_image: np.ndarray,
                fixed_image: np.ndarray,
                method: AlignmentMethod = AlignmentMethod.PHASE_CORRELATION) -> AlignmentResult:
    """Convenience function for single image alignment."""
    aligner = UnifiedAligner()
    return aligner.align_images(moving_image, fixed_image, method)


def align_stack(image_stack: np.ndarray,
               reference_strategy: ReferenceStrategy = ReferenceStrategy.PREVIOUS_IMAGE) -> StackAlignmentResult:
    """Convenience function for stack alignment."""
    aligner = UnifiedAligner()
    return aligner.align_stack(image_stack, reference_strategy)
