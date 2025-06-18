# Contributing to IQID-Alphas

We welcome contributions to the IQID-Alphas project! This guide will help you get started with contributing code, documentation, or bug reports.

## 📋 Table of Contents

- [Getting Started](#getting-started6. **Test Your Changes**
   ```bash
   python -m unittest discover iqid_alphas.tests -v
   ```[Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of image processing and scientific computing

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/iqid-alphas.git
   cd iqid-alphas
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in development mode
   ```

4. **Run Tests to Verify Setup**
   ```bash
   python -m unittest discover iqid_alphas.tests -v
   ```

## 📝 Contribution Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Identify and fix issues in the codebase
- **Feature enhancements**: Add new functionality or improve existing features
- **Documentation**: Improve documentation, examples, or tutorials
- **Testing**: Add or improve test coverage
- **Performance optimizations**: Improve processing speed or memory usage

### Before You Start

1. **Check existing issues**: Look for existing issues or discussions related to your contribution
2. **Create an issue**: For new features or major changes, create an issue to discuss the approach
3. **Keep it focused**: Make one logical change per pull request

## 🔧 Code Standards

### Python Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add type hints where appropriate
- Maximum line length: 100 characters

### Code Organization

- **Core modules** (`iqid_alphas/core/`): Fundamental image processing operations
- **Pipelines** (`iqid_alphas/pipelines/`): High-level workflow orchestration
- **Utils** (`iqid_alphas/utils/`): Shared utility functions
- **Visualization** (`iqid_alphas/visualization/`): Plotting and display functions

### Configuration-Driven Design

- All processing parameters should be configurable via JSON files
- Use the established configuration patterns in `configs/`
- Avoid hard-coding parameters in the source code

### Error Handling

- Use descriptive error messages
- Include context about what operation failed
- Log important events and errors appropriately

### Example Code Style

```python
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

class ImageProcessor:
    """
    Process iQID images with configurable parameters.
    
    This class provides methods for processing quantitative imaging data
    with support for various image formats and processing options.
    
    Args:
        config: Configuration dictionary or path to config file
        verbose: Enable verbose logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        self.config = config or {}
        self.verbose = verbose
        
    def process_image(self, image_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single iQID image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (processed_image, metadata)
            
        Raises:
            ValueError: If image file is not found or invalid
            ProcessingError: If processing fails
        """
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")
            
        # Implementation here...
        return processed_image, metadata
```

## 🧪 Testing

### Test Structure

Our test suite is organized into:

- **Unit tests** (`iqid_alphas/tests/core/`, `iqid_alphas/tests/pipelines/`): Test individual components
- **Integration tests** (`iqid_alphas/tests/integration/`): Test complete workflows
- **Performance tests** (`evaluation/`): Benchmark processing performance

### Writing Tests

1. **Use pytest**: All tests should use the pytest framework
2. **Test file naming**: Test files should start with `test_`
3. **Descriptive test names**: Use clear, descriptive test function names
4. **Mock external dependencies**: Use mocks for file I/O and external services
5. **Test edge cases**: Include tests for error conditions and edge cases

### Example Test

```python
import pytest
import numpy as np
from iqid_alphas.core.processor import IQIDProcessor

class TestIQIDProcessor:
    """Test cases for IQIDProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = IQIDProcessor()
        self.sample_image = np.random.rand(512, 512).astype(np.float32)
    
    def test_process_valid_image(self):
        """Test processing with valid input image."""
        result = self.processor.process(self.sample_image)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == self.sample_image.shape
    
    def test_process_invalid_input(self):
        """Test processing with invalid input."""
        with pytest.raises(ValueError, match="Invalid input"):
            self.processor.process(None)
```

### Running Tests

```bash
# Run all tests
python -m unittest discover iqid_alphas.tests -v

# Run CLI tests specifically
python -m unittest iqid_alphas.tests.test_cli -v

# Run integration tests
python -m unittest iqid_alphas.tests.integration.test_consolidated_workflow -v

# Run core module tests
python -m unittest iqid_alphas.tests.core -v
```

## 📚 Documentation

### Types of Documentation

1. **API Documentation**: Docstrings in the code (following Google style)
2. **User Guides**: Step-by-step tutorials in `docs/user_guides/`
3. **Examples**: Working code examples in `examples/`
4. **Technical Documentation**: Design documents in `docs/technical/`

### Writing Documentation

- Use clear, concise language
- Include code examples where appropriate
- Update documentation when making code changes
- Follow the existing documentation structure

### Generating API Documentation

```bash
# Generate API documentation (if using Sphinx)
cd docs/
make html
```

## 🔄 Submitting Changes

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the style guidelines
   - Add or update tests as needed
   - Update documentation if necessary

3. **Test your changes**
   ```bash
   python -m pytest iqid_alphas/tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: descriptive commit message"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a pull request**
   - Use a descriptive title
   - Include a detailed description of the changes
   - Reference any related issues
   - Ensure all tests pass

### Pull Request Guidelines

- **Keep PRs focused**: One logical change per pull request
- **Update tests**: Add tests for new functionality
- **Update documentation**: Include documentation updates
- **Add changelog entry**: Update CHANGELOG.md for significant changes
- **Be responsive**: Respond to review feedback promptly

### Commit Message Format

Use clear, descriptive commit messages:

```
Add feature: Brief description of the change

Longer description if needed, explaining:
- What was changed
- Why it was changed
- Any special considerations

Closes #123
```

## 🐛 Reporting Issues

When reporting bugs, please include:

- **System information**: OS, Python version, package versions
- **Steps to reproduce**: Clear steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Full error messages and stack traces
- **Sample data**: Minimal example that reproduces the issue (if possible)

## 💬 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check the `docs/` directory for detailed guides

## 📜 License

By contributing to IQID-Alphas, you agree that your contributions will be licensed under the same license as the project (see LICENSE.txt).

## 🙏 Acknowledgments

Thank you for contributing to IQID-Alphas! Your contributions help make this project better for the entire scientific community.
