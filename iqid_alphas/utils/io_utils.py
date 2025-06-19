"""
I/O and File Management Utilities

Migrated and refactored from old/iqid-alphas/iqid/helper.py
Contains file system operations, directory management, and data loading utilities.
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
from tqdm import trange


def list_directories(root_dir: Union[str, Path]) -> List[str]:
    """Get list of directory paths for all folders in the current directory.
    
    Migrated from helper.list_studies()
    
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
    """Get list of directory paths for all subfolders one level down.
    
    Migrated from helper.list_substudies()
    
    Parameters
    ----------
    root_dir : str or Path
        Root directory to search
        
    Returns
    -------
    List[str]
        List of subdirectory paths
    """
    study_list = list_directories(root_dir)
    substudy_list = []
    for study in study_list:
        substudies = list_directories(study)
        substudy_list.extend(substudies)
    return substudy_list


def natural_sort_key(text: str) -> List[Union[int, str]]:
    """Generate sort key for natural sorting (handles numbers in strings).
    
    Migrated from helper.natural_keys()
    
    Parameters
    ----------
    text : str
        Text to generate sort key for
        
    Returns
    -------
    List[Union[int, str]]
        Sort key for natural ordering
    """
    def _atoi(text: str) -> Union[int, str]:
        return int(text) if text.isdigit() else text
    
    return [_atoi(c) for c in re.split(r'(\d+)', text)]


def natural_sort(items: List[str]) -> List[str]:
    """Sort list of strings in natural order (handles numbers correctly).
    
    Migrated from helper.natural_sort()
    
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


def find_files_with_pattern(directory: Union[str, Path], 
                           pattern: str, 
                           recursive: bool = False) -> List[Path]:
    """Find files matching a glob pattern in directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search
    pattern : str
        Glob pattern to match
    recursive : bool, default False
        Whether to search recursively
        
    Returns
    -------
    List[Path]
        List of matching file paths
    """
    directory = Path(directory)
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Parameters
    ----------
    directory : str or Path
        Directory path to ensure exists
        
    Returns
    -------
    Path
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with file information
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = file_path.stat()
    return {
        'name': file_path.name,
        'stem': file_path.stem,
        'suffix': file_path.suffix,
        'parent': str(file_path.parent),
        'size_bytes': stat.st_size,
        'modified_time': stat.st_mtime,
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir()
    }


def copy_file_safe(source: Union[str, Path], 
                   destination: Union[str, Path], 
                   create_dirs: bool = True) -> Path:
    """Safely copy a file to destination, creating directories if needed.
    
    Parameters
    ----------
    source : str or Path
        Source file path
    destination : str or Path
        Destination file path
    create_dirs : bool, default True
        Whether to create destination directories
        
    Returns
    -------
    Path
        Path to the copied file
    """
    import shutil
    
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    
    if create_dirs:
        destination.parent.mkdir(parents=True, exist_ok=True)
    
    return Path(shutil.copy2(source, destination))


def validate_file_extension(file_path: Union[str, Path], 
                          allowed_extensions: List[str]) -> bool:
    """Validate that file has one of the allowed extensions.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
    allowed_extensions : List[str]
        List of allowed extensions (with or without dots)
        
    Returns
    -------
    bool
        True if extension is allowed, False otherwise
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    # Normalize extensions to include dots
    normalized_extensions = []
    for ext in allowed_extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized_extensions.append(ext.lower())
    
    return extension in normalized_extensions


def load_text_file(file_path: Union[str, Path], 
                   encoding: str = 'utf-8') -> str:
    """Load text file content.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the text file
    encoding : str, default 'utf-8'
        File encoding
        
    Returns
    -------
    str
        File content
    """
    file_path = Path(file_path)
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def save_text_file(content: str, 
                   file_path: Union[str, Path], 
                   encoding: str = 'utf-8',
                   create_dirs: bool = True) -> Path:
    """Save text content to file.
    
    Parameters
    ----------
    content : str
        Text content to save
    file_path : str or Path
        Path to save the file
    encoding : str, default 'utf-8'
        File encoding
    create_dirs : bool, default True
        Whether to create directories if they don't exist
        
    Returns
    -------
    Path
        Path to the saved file
    """
    file_path = Path(file_path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)
    
    return file_path


def load_image(file_path: Union[str, Path], 
               color_mode: str = 'unchanged') -> np.ndarray:
    """
    Load an image from file using OpenCV.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the image file
    color_mode : str
        Color mode for loading:
        - 'unchanged': Load as-is (cv2.IMREAD_UNCHANGED)
        - 'color': Load as color image (cv2.IMREAD_COLOR)
        - 'grayscale': Load as grayscale (cv2.IMREAD_GRAYSCALE)
    
    Returns
    -------
    np.ndarray
        Loaded image as numpy array
        
    Raises
    ------
    ValueError
        If file cannot be loaded or does not exist
    FileNotFoundError
        If file does not exist
    """
    import cv2
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Map color mode to OpenCV flags
    mode_map = {
        'unchanged': cv2.IMREAD_UNCHANGED,
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE
    }
    
    if color_mode not in mode_map:
        raise ValueError(f"Invalid color_mode: {color_mode}. "
                        f"Must be one of {list(mode_map.keys())}")
    
    # Load image
    image = cv2.imread(str(file_path), mode_map[color_mode])
    
    if image is None:
        raise ValueError(f"Could not load image from {file_path}. "
                        "File may be corrupted or in unsupported format.")
    
    return image


def save_image(image: np.ndarray,
               file_path: Union[str, Path],
               create_dirs: bool = True) -> Path:
    """
    Save an image to file using OpenCV.
    
    Parameters
    ----------
    image : np.ndarray
        Image array to save
    file_path : str or Path
        Path where to save the image
    create_dirs : bool
        Whether to create parent directories if they don't exist
        
    Returns
    -------
    Path
        Path object of the saved file
        
    Raises
    ------
    ValueError
        If image cannot be saved
    """
    import cv2
    
    file_path = Path(file_path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save image
    success = cv2.imwrite(str(file_path), image)
    
    if not success:
        raise ValueError(f"Could not save image to {file_path}")
    
    return file_path
