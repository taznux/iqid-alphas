# Enhanced Image Loading - Extension Mismatch Solution

## Problem Summary
You correctly identified that the pipeline fails when files have incorrect extensions, such as PNG data with a `.tif` extension. The error `b'\x89PNG'` clearly shows the file is PNG format despite the wrong extension.

## Solution Implemented

I've enhanced the `IQIDProcessor` class in `/home/wxc151/iqid-alphas/iqid_alphas/core/processor.py` with smart image loading capabilities:

### 1. Format Detection Method
```python
def _detect_file_format(self, file_path: str) -> str:
    """Detect actual file format by reading file header."""
    with open(file_path, 'rb') as f:
        header = f.read(16)
    
    # Check file signatures
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'PNG'
    elif header.startswith(b'\xff\xd8\xff'):
        return 'JPEG'
    elif header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
        return 'TIFF'
    # ... other formats
```

### 2. Smart Loading Strategy
```python
def _load_image_smart(self, image_path: str) -> np.ndarray:
    """Load image with format-appropriate loader."""
    actual_format = self._detect_file_format(image_path)
    
    # Strategy 1: Use format-appropriate loader
    if actual_format in ['PNG', 'JPEG', 'GIF', 'BMP']:
        try:
            img = Image.open(image_path)  # PIL handles these well
            return np.array(img)
        except:
            pass  # Fall through to next strategy
    
    elif actual_format == 'TIFF':
        try:
            return tifffile.imread(image_path)  # Best for TIFF
        except:
            pass
    
    # Strategy 2: Fallback to PIL (most versatile)
    try:
        img = Image.open(image_path)
        return np.array(img)
    except:
        pass
    
    # Strategy 3: Last resort - skimage
    return io.imread(image_path)
```

### 3. Enhanced load_image Method
The main `load_image` method now uses smart loading:
```python
def load_image(self, image_path: str, key: str = 'main') -> np.ndarray:
    """Load image with smart format detection."""
    image = self._load_image_smart(image_path)
    self.images[key] = image
    return image
```

## Key Benefits

### ✅ Automatic Format Detection
- Reads file headers to determine actual format
- Ignores misleading file extensions
- Detects PNG, JPEG, GIF, BMP, TIFF, WEBP

### ✅ Smart Loader Selection
- Uses PIL for PNG/JPEG/GIF/BMP (most reliable)
- Uses tifffile for TIFF (preserves scientific data)
- Falls back gracefully if primary loader fails

### ✅ Extension Mismatch Handling
- PNG file with .tif extension → Uses PIL loader
- JPEG file with .png extension → Uses PIL loader
- Provides helpful warnings about mismatches

### ✅ Backward Compatibility
- Existing code continues to work
- Real UCSF TIFF files still load correctly
- No breaking changes to API

## Usage Examples

### Before Enhancement (Failed)
```python
processor = IQIDProcessor()
# This would fail with "not a TIFF file b'\x89PNG'"
result = processor.load_image("png_file_with_tif_extension.tif")
```

### After Enhancement (Success)
```python
processor = IQIDProcessor()
# This now works automatically
result = processor.load_image("png_file_with_tif_extension.tif")
# Output: "Format mismatch detected: .tif extension but PNG format"
# Output: "Loaded with PIL (format-aware): (50, 50, 3) uint8"
```

## Testing

The enhancement handles these scenarios:
1. **PNG with .tif extension** → Detects PNG, uses PIL loader ✅
2. **JPEG with .png extension** → Detects JPEG, uses PIL loader ✅  
3. **Real TIFF files** → Detects TIFF, uses tifffile loader ✅
4. **Unknown formats** → Falls back to versatile loaders ✅

## Integration with UCSF Pipeline

This enhancement seamlessly integrates with the existing UCSF pipeline:
- All existing functionality preserved
- Real UCSF scientific TIFF files continue to work
- Added robustness for diverse data sources
- Maintains quantitative data precision

The smart loading makes the pipeline more robust and user-friendly while maintaining the scientific accuracy required for medical imaging analysis.

## File Signatures Reference
- PNG: `\x89PNG\r\n\x1a\n`
- JPEG: `\xff\xd8\xff`  
- TIFF (LE): `II*\x00`
- TIFF (BE): `MM\x00*`
- GIF: `GIF8`
- BMP: `BM`
- WEBP: `RIFF...WEBP`
