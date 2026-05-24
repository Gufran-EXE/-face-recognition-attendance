# 🤝 Contributing to Face Recognition Attendance System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on the code, not the person
- Help others learn and grow
- Report harassment or inappropriate behavior

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/-face-recognition-attendance.git
cd -face-recognition-attendance

# Add upstream
git remote add upstream https://github.com/Gufran-EXE/-face-recognition-attendance.git
```

### 2. Create Feature Branch

```bash
# Update main branch
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
# or
git checkout -b docs/documentation-update
```

### 3. Set Up Development Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
```

## Development Workflow

### Code Style

We follow:
- **Python:** [PEP 8](https://www.python.org/dev/peps/pep-0008/) via Black
- **React:** [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript)

#### Python Formatting

```bash
# Format code with Black
black src/ backend/ tests/

# Sort imports with isort
isort src/ backend/ tests/

# Lint with pylint
pylint src/ backend/

# Type checking
mypy src/ backend/
```

#### Pre-commit Hooks

```bash
# Automatically run before commit
pre-commit run --all-files

# Skip pre-commit (not recommended)
git commit --no-verify
```

### Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_face_service.py -v

# Watch mode
pytest-watch
```

**Coverage Requirements:**
- Minimum 80% code coverage
- All public functions must have tests
- All bug fixes must include tests

### Git Commits

**Commit Message Format:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Code style changes (formatting, missing semicolons, etc.)
- `refactor:` Code refactoring without feature changes
- `perf:` Performance improvements
- `test:` Adding or updating tests
- `chore:` Build, dependencies, tooling
- `ci:` CI/CD configuration

**Examples:**

```
feat(auth): implement JWT token refresh

Implement automatic JWT token refresh mechanism to improve UX.
Users no longer need to login when token expires within session.

Fixes #123
```

```
fix(face-recognition): fix duplicate entry bug

Prevent same face from being logged multiple times in 5 minute window.
Added timestamp check before marking attendance.

Fixes #456
```

## Pull Request Process

### 1. Before Submitting PR

```bash
# Update from upstream
git fetch upstream
git rebase upstream/main

# Run all checks
black src/ backend/ tests/
isort src/ backend/ tests/
pylint src/ backend/
mypy src/ backend/
pytest --cov=src

# Commit and push
git push origin feature/your-feature-name
```

### 2. Create Pull Request

**PR Title Format:**
```
[Type] Brief description - Issue #123
```

Examples:
- `[Feature] Add face encoding caching - Issue #456`
- `[Fix] Resolve duplicate attendance logging - Issue #789`
- `[Docs] Update API documentation`

**PR Description Template:**

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Documentation update
- [ ] Refactoring

## Related Issues
Fixes #(issue number)

## Changes Made
- List major changes
- Use bullet points

## Testing
- [ ] Added tests
- [ ] All tests pass
- [ ] Coverage >= 80%

## Screenshots (if applicable)
Add screenshots for UI changes.

## Checklist
- [ ] Code follows style guidelines
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Added tests for new features
- [ ] All tests pass locally
```

### 3. Review Process

Reviewers will check:
- ✅ Code quality and style
- ✅ Test coverage
- ✅ Documentation
- ✅ No breaking changes
- ✅ Security implications

### 4. Merge

Once approved:
```bash
# Rebase and merge
git rebase upstream/main
git push origin feature/your-feature-name
```

PR will be merged to `main` branch.

## Documentation Requirements

### Python Code

```python
\"\"\"Module-level docstring.\"\"\"

def function_name(param1: str, param2: int) -> bool:
    \"\"\"
    Brief description of function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When validation fails
        ConnectionError: When DB connection fails

    Example:
        >>> result = function_name(\"test\", 42)
        >>> result
        True
    \"\"\"
    pass
```

### React Components

```jsx
/**
 * Brief component description.
 *
 * Longer description with usage details.
 *
 * @component
 * @example
 * const args = {
 *   title: 'Example',
 *   onSubmit: (data) => console.log(data)
 * }
 * return <Form {...args} />
 */
export function Form({ title, onSubmit }) {
  // Component implementation
}
```

## Questions or Issues?

- 💬 Ask in GitHub Discussions
- 🐛 Report bugs in Issues
- 📧 Email maintainers for security issues
- 📖 Check documentation first

---

**Thank you for contributing! 🎉**
