"""
I/O and File Management Utilities

Essential file system operations, directory management, and logging utilities
for the IQID-Alphas pipeline system.
"""

import os
import logging
import sys
import re
import glob
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np


def setup_logging(level: str = "INFO", 
                 log_file: Optional[Union[str, Path]] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Parameters
    ----------
    level : str, default "INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str or Path, optional
        Path to log file. If None, logs only to console
    format_string : str, optional
        Custom log format string
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    logger = logging.getLogger('iqid_alphas')
    logger.handlers.clear()  # Clear any existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    directory : str or Path
        Directory path to create
        
    Returns
    -------
    Path
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def list_directories(root_dir: Union[str, Path]) -> List[str]:
    """
    Get list of directory paths for all folders in the current directory.
    
    Parameters
    ----------
    root_dir : str or Path
        Root directory to search
        
    Returns
    -------
    List[str]
        List of directory paths
    """
    root_dir = Path(root_dir)
    return [str(f) for f in root_dir.iterdir() if f.is_dir()]


def list_subdirectories(root_dir: Union[str, Path]) -> List[str]:
    """
    Get list of directory paths for all subfolders one level down.
    
    Parameters
    ----------
    root_dir : str or Path
        Root directory to search
        
    Returns
    -------
    List[str]
        List of subdirectory paths
    """
    root_dir = Path(root_dir)
    subdirs = []
    for item in root_dir.iterdir():
        if item.is_dir():
            subdirs.extend([str(subitem) for subitem in item.iterdir() if subitem.is_dir()])
    return subdirs


def natural_sort_key(text: str) -> List[Union[int, str]]:
    """
    Generate sort key for natural sorting (handles numbers properly).
    
    Parameters
    ----------
    text : str
        Text to generate sort key for
        
    Returns
    -------
    List[Union[int, str]]
        Sort key that handles numbers naturally
    """
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s
    
    return [tryint(c) for c in re.split(r'(\d+)', text)]


def natural_sort(items: List[str]) -> List[str]:
    """
    Sort a list of strings naturally (handles numbers properly).
    
    Parameters
    ----------
    items : List[str]
        List of strings to sort
        
    Returns
    -------
    List[str]
        Naturally sorted list
    """
    return sorted(items, key=natural_sort_key)


def find_files_with_pattern(directory: Union[str, Path], pattern: str) -> List[str]:
    """
    Find files matching a pattern in a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search
    pattern : str
        File pattern to match
        
    Returns
    -------
    List[str]
        List of matching file paths
    """
    search_pattern = str(Path(directory) / pattern)
    return glob.glob(search_pattern)


def find_files_by_pattern(root_dir: Union[str, Path], 
                         pattern: str,
                         recursive: bool = True) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Parameters
    ----------
    root_dir : str or Path
        Root directory to search
    pattern : str
        Glob pattern to match (e.g., "*.tif", "raw/*.tiff")
    recursive : bool, default True
        Whether to search recursively
        
    Returns
    -------
    List[Path]
        List of matching file paths
    """
    root_dir = Path(root_dir)
    
    if recursive:
        return list(root_dir.rglob(pattern))
    else:
        return list(root_dir.glob(pattern))


def copy_file_safe(src: Union[str, Path], dst: Union[str, Path], 
                  create_dirs: bool = True) -> Path:
    """
    Safely copy a file to destination.
    
    Parameters
    ----------
    src : str or Path
        Source file path
    dst : str or Path
        Destination file path
    create_dirs : bool, default True
        Whether to create destination directories
        
    Returns
    -------
    Path
        Destination file path
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if create_dirs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src_path, dst_path)
    return dst_path


def load_image(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image file as numpy array.
    
    Parameters
    ----------
    file_path : str or Path
        Path to image file
        
    Returns
    -------
    np.ndarray
        Image array
    """
    try:
        import cv2
        return cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    except ImportError:
        # Fallback to other image loading libraries
        try:
            from PIL import Image
            return np.array(Image.open(file_path))
        except ImportError:
            raise ImportError("No image loading library available (cv2 or PIL)")


def save_image(image: np.ndarray, 
              file_path: Union[str, Path],
              create_dirs: bool = True) -> Path:
    """
    Save an image array to file.
    
    Parameters
    ----------
    image : np.ndarray
        Image array to save
    file_path : str or Path
        Output file path
    create_dirs : bool, default True
        Whether to create parent directories
        
    Returns
    -------
    Path
        Path to saved file
    """
    file_path = Path(file_path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import cv2
        success = cv2.imwrite(str(file_path), image)
        if not success:
            raise ValueError(f"Could not save image to {file_path}")
    except ImportError:
        # Fallback to PIL
        try:
            from PIL import Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            Image.fromarray(image).save(file_path)
        except ImportError:
            raise ImportError("No image saving library available (cv2 or PIL)")
    
    return file_path


def save_json(data: Dict[str, Any], 
              file_path: Union[str, Path],
              create_dirs: bool = True) -> Path:
    """
    Save data as JSON file.
    
    Parameters
    ----------
    data : dict
        Data to save
    file_path : str or Path
        Output file path
    create_dirs : bool, default True
        Whether to create parent directories
        
    Returns
    -------
    Path
        Path to saved file
    """
    import json
    
    file_path = Path(file_path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_serializer)
    
    return file_path


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Parameters
    ----------
    file_path : str or Path
        Input file path
        
    Returns
    -------
    dict
        Loaded data
    """
    import json
    
    with open(file_path, 'r') as f:
        return json.load(f)


def _json_serializer(obj):
    """JSON serializer for numpy types and Path objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def is_valid_image_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is a valid image file based on extension.
    
    Parameters
    ----------
    file_path : str or Path
        File path to check
        
    Returns
    -------
    bool
        True if file appears to be an image
    """
    valid_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    return Path(file_path).suffix.lower() in valid_extensions
