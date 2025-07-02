# Contributing to Thai Text Classification Project

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use the issue template** when creating new issues
3. **Provide detailed information** including:
   - Python version
   - Dependencies versions
   - Error messages and stack traces
   - Steps to reproduce

### Suggesting Features

1. **Check the roadmap** to see if it's already planned
2. **Create a feature request** with:
   - Clear description of the feature
   - Use case and motivation
   - Proposed implementation approach

### Code Contributions

#### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/Text-Classification-Models-Thai.git
cd Text-Classification-Models-Thai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards
   - Add tests for new functionality
   - Update documentation

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Check code quality**
   ```bash
   black src/ examples/ utils/
   flake8 src/ examples/ utils/
   mypy src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

#### Commit Message Convention

We use conventional commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code formatting
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Examples:
```
feat: add multi-class classification support
fix: resolve tokenization issue with Thai text
docs: update installation instructions
```

#### Code Standards

1. **Python Style**
   - Follow PEP 8
   - Use Black for formatting
   - Maximum line length: 88 characters
   - Use type hints

2. **Documentation**
   - Write docstrings for all functions and classes
   - Use Google-style docstrings
   - Update README.md for user-facing changes

3. **Testing**
   - Write unit tests for new functions
   - Maintain test coverage above 80%
   - Use pytest for testing

#### File Structure Guidelines

```
src/
├── models/          # Model architectures
├── data/           # Data processing
├── training/       # Training utilities
└── inference/      # Inference utilities

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── fixtures/       # Test data

examples/
├── basic/          # Simple examples
└── advanced/       # Complex examples
```

## 📝 Code Review Process

1. **All contributions** require code review
2. **Reviewers check for**:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Performance implications

3. **Address feedback** promptly and professionally
4. **Squash commits** before merging if requested

## 🧪 Testing Guidelines

### Unit Tests
```python
# tests/test_pipeline.py
import pytest
from src.text_classification_pipeline import TextClassificationPipeline

def test_pipeline_initialization():
    pipeline = TextClassificationPipeline()
    assert pipeline.model_name is not None

def test_text_preprocessing():
    pipeline = TextClassificationPipeline()
    result = pipeline.preprocess_text("  test text  ")
    assert result == "test text"
```

### Integration Tests
```python
# tests/test_integration.py
def test_full_pipeline():
    # Test complete pipeline workflow
    pass
```

### Performance Tests
```python
# tests/test_performance.py
def test_inference_speed():
    # Test inference performance
    pass
```

## 📚 Documentation Guidelines

1. **API Documentation**
   - Document all public functions
   - Include examples in docstrings
   - Keep documentation up-to-date

2. **User Documentation**
   - Update README.md for user-facing changes
   - Add examples for new features
   - Maintain the notebook tutorials

3. **Developer Documentation**
   - Document complex algorithms
   - Explain design decisions
   - Add architecture diagrams when helpful

## 🎯 Areas for Contribution

### High Priority
- [ ] Additional Thai language models support
- [ ] Performance optimizations
- [ ] More evaluation metrics
- [ ] Data augmentation techniques

### Medium Priority
- [ ] Multi-language support
- [ ] Advanced hyperparameter tuning
- [ ] Model compression techniques
- [ ] Visualization improvements

### Low Priority
- [ ] Additional deployment options
- [ ] Integration with more frameworks
- [ ] Advanced monitoring features

## 🚀 Release Process

1. **Version numbering**: We use semantic versioning (MAJOR.MINOR.PATCH)
2. **Release branches**: Create release branches for major/minor releases
3. **Changelog**: Update CHANGELOG.md with all changes
4. **Testing**: All tests must pass before release
5. **Documentation**: Update version-specific documentation

## 🤔 Questions?

- **General questions**: Open a discussion
- **Bug reports**: Create an issue
- **Feature requests**: Create an issue with feature template
- **Security issues**: Email maintainers privately

## 🏆 Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to the Thai Text Classification project! 🙏
