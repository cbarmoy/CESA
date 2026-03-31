# Contributing to CESA

Thank you for your interest in contributing to CESA (Complex EEG Studio Analysis)!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/CESA.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes locally
6. Commit your changes: `git commit -m "Add your commit message"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Verify imports
python -c "import CESA; import core; print('✅ All modules imported successfully')"

# Run the application
python run.py
```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and small

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in imperative mood (e.g., "Add", "Fix", "Update")
- Reference issues when applicable: "Fix #123: description"

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Ensure all tests pass
3. Update documentation as needed
4. Request review from maintainers

## Reporting Bugs

Please use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) when reporting issues.

## Feature Requests

Please use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) for new feature suggestions.

## Questions?

Feel free to open an issue for any questions or concerns.

