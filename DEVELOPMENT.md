# Development Setup Guide

This guide will help you set up the development environment for Healthcare AI Recommendation System.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [Common Issues](#common-issues)
5. [Development Workflow](#development-workflow)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended for ML models)
- **Disk Space**: 2GB
- **Internet**: Required for package installation

### Software Requirements
- Git
- Python 3.8+
- pip (Python package manager)
- Virtual environment tool (venv or virtualenv)
- Text editor or IDE (VS Code, PyCharm, etc.)

---

## Installation Steps

### Step 1: Clone Repository
```bash
# SSH
git clone git@github.com:YOUR-USERNAME/medi_recomm.git

# HTTPS
git clone https://github.com/YOUR-USERNAME/medi_recomm.git

# Navigate to project
cd medi_recomm
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Verify activation (should show (venv) in terminal)
```

### Step 3: Upgrade pip
```bash
# Windows
python -m pip install --upgrade pip

# macOS/Linux
python3 -m pip install --upgrade pip
```

### Step 4: Install Dependencies
```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies (optional but recommended)
pip install -r requirements-dev.txt
```

### Step 5: Prepare Data Directory
```bash
# Data files should be in Data/ directory
# Ensure these files exist:
# - Data/dataset.csv
# - Data/Drug.csv
# - Data/users.db (created on first run)

# If files don't exist, you can create empty ones or download sample data
```

### Step 6: Configure Environment (Optional)
```bash
# Copy template
cp .env.example .env

# Edit .env with your settings
# On Windows
notepad .env

# On macOS/Linux
nano .env
```

---

## Verification

### Verify Python Installation
```bash
python --version
# Should output: Python 3.8.x or higher
```

### Verify Virtual Environment
```bash
# Windows
where python
# Should show path in venv/Scripts

# macOS/Linux
which python
# Should show path in venv/bin
```

### Verify Dependencies
```bash
pip list
# Should show all installed packages including streamlit, pandas, scikit-learn, etc.
```

### Test Application Launch
```bash
streamlit run app.py
# Application should open at http://localhost:8501
```

---

## Common Issues

### Issue: "Python command not found"
**Solution:**
- Verify Python installation: `python --version`
- Add Python to PATH environment variable
- Restart terminal after installation

### Issue: Virtual Environment not activating
**Solution (Windows):**
```bash
# Try alternative activation script
venv\Scripts\activate.bat
# or
venv\Scripts\activate.ps1
```

**Solution (macOS/Linux):**
```bash
# Make script executable
chmod +x venv/bin/activate
source venv/bin/activate
```

### Issue: Package installation fails
**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Try installing with specific Python version
python3.9 -m pip install -r requirements.txt

# Install with verbose output to see error
pip install -v package-name
```

### Issue: Streamlit port already in use
**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill process using port 8501
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8501
kill -9 <PID>
```

### Issue: SQLite database locked
**Solution:**
```bash
# Remove old database and let it recreate
rm Data/users.db
# or on Windows
del Data\users.db

# Restart application
streamlit run app.py
```

### Issue: Module import errors
**Solution:**
```bash
# Ensure virtual environment is activated
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

---

## Development Workflow

### 1. Before Starting Work
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Update from upstream
git fetch upstream
git checkout main
git merge upstream/main
```

### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. During Development
```bash
# Run application
streamlit run app.py

# Run tests (if added)
pytest

# Check code style
flake8 .

# Format code
black .
```

### 4. Before Committing
```bash
# Check changes
git status
git diff

# Stage changes
git add .

# Commit with message
git commit -m "feat: add new feature description"

# Review commit
git log -1 --name-status
```

### 5. Push and Create PR
```bash
# Push branch
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

---

## IDE Setup Recommendations

### Visual Studio Code
1. Install Python extension
2. Select interpreter from venv
3. Install optional extensions:
   - Pylance (Python language server)
   - Python Docstring Generator
   - Streamlit

### PyCharm
1. Create project from existing sources
2. Set interpreter to venv/bin/python
3. Enable code inspections
4. Install Streamlit plugin

### Settings
```json
// .vscode/settings.json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=99"],
    "[python]": {
        "editor.defaultFormatter": "ms-python.python",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

---

## Useful Development Commands

```bash
# Run linter
flake8 . --max-line-length=99

# Format code
black . --line-length=99

# Sort imports
isort .

# Run type checker
mypy .

# Create dependency list
pip freeze > requirements.txt

# Check for security issues
bandit -r .

# Create distribution
python setup.py sdist bdist_wheel

# Upload to PyPI (if publishing)
twine upload dist/*
```

---

## Performance Optimization

### For Development
```python
# Use caching in Streamlit
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')

# Use caching for ML models
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')
```

### For Production
- Enable production mode in Streamlit config
- Use smaller model versions
- Implement database connection pooling
- Add CDN for static assets

---

## Debugging Tips

### Streamlit Debugging
```bash
# Run with logger output
streamlit run app.py --logger.level=debug

# Check Streamlit cache
streamlit cache clear
```

### Python Debugging
```python
import pdb

# Set breakpoint (Python 3.7+)
breakpoint()

# Or use pdb
pdb.set_trace()

# Commands: n (next), s (step), c (continue), l (list), p (print)
```

### Check Logs
```bash
# View Streamlit logs
tail -f .streamlit/logger.log

# View application logs
# Check app.log file in project directory
```

---

## Next Steps

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
2. Review [README.md](README.md) for project overview
3. Check existing [GitHub Issues](../../issues) for tasks
4. Start with "good first issue" labels
5. Ask questions in GitHub Discussions

---

## Support

Need help? 
- 📧 Email: support@healthai.com
- 💬 GitHub Discussions: [Ask questions](../../discussions)
- 🐛 GitHub Issues: [Report bugs](../../issues)

---

**Last Updated**: January 2026  
**Version**: 1.0.0
