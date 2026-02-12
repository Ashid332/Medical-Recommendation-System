# Troubleshooting Guide

## Common Issues and Solutions

### Table of Contents
1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Performance Issues](#performance-issues)
4. [Database Issues](#database-issues)
5. [Authentication Issues](#authentication-issues)
6. [ML Model Issues](#ml-model-issues)
7. [Data Issues](#data-issues)

---

## Installation Issues

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Symptoms**: ImportError when running application

**Solutions**:
```bash
# 1. Verify virtual environment is activated
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import streamlit; print(streamlit.__version__)"

# 4. If still failing, reinstall
pip uninstall streamlit -y
pip install streamlit==1.28.0
```

### Issue: "pip is not recognized as an internal or external command"
**Symptoms**: Command not found when installing packages

**Solutions**:
```bash
# 1. Use Python module directly
python -m pip install -r requirements.txt

# 2. Add Python to PATH (Windows)
# Control Panel > Environment Variables > Add Python\Scripts to PATH

# 3. Use Anaconda if available
conda install --file requirements.txt
```

### Issue: "Python version is too old"
**Symptoms**: Python version < 3.8

**Solutions**:
```bash
# 1. Check current version
python --version

# 2. Install new version from python.org
# Download Python 3.9+ from https://www.python.org/

# 3. Verify new version
python3.9 --version

# 4. Create virtual environment with new version
python3.9 -m venv venv
```

### Issue: "Permission denied: venv/bin/activate"
**Symptoms**: Cannot activate virtual environment (macOS/Linux)

**Solutions**:
```bash
# Make script executable
chmod +x venv/bin/activate

# Try activation again
source venv/bin/activate

# Alternative: Use Python module
python -m venv venv --upgrade-deps
```

---

## Runtime Errors

### Issue: "FileNotFoundError: Data/dataset.csv not found"
**Symptoms**: Application fails to load dataset

**Solutions**:
```bash
# 1. Check file exists
ls Data/  # macOS/Linux
dir Data  # Windows

# 2. Create Data directory if missing
mkdir Data  # macOS/Linux
mkdir Data  # Windows

# 3. Copy dataset files to Data/
# Ensure these files exist:
# - Data/dataset.csv
# - Data/Drug.csv

# 4. Check file permissions
chmod 644 Data/*.csv  # macOS/Linux
```

### Issue: "ImportError: cannot import name 'option_menu'"
**Symptoms**: streamlit-option-menu not installed

**Solutions**:
```bash
# 1. Install missing package
pip install streamlit-option-menu

# 2. Verify installation
python -c "from streamlit_option_menu import option_menu; print('OK')"

# 3. Check version compatibility
pip list | grep streamlit
```

### Issue: "TypeError: unhashable type"
**Symptoms**: Session state key error

**Solutions**:
```python
# Wrong (unhashable dict)
st.session_state[{'key': 'value'}] = something

# Correct (use string key)
st.session_state['my_key'] = something
```

### Issue: "AttributeError: module 'X' has no attribute 'Y'"
**Symptoms**: Function/class not found in module

**Solutions**:
```bash
# 1. Check library version
pip show package-name

# 2. Update to compatible version
pip install package-name==X.X.X

# 3. Check imports
python -c "from package import function; print(dir(function))"
```

---

## Performance Issues

### Issue: "Application is very slow to load"
**Symptoms**: Long startup time (>10 seconds)

**Solutions**:
```python
# Use caching to speed up
@st.cache_data
def load_data():
    return pd.read_csv('Data/dataset.csv')

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

# Check what's slow
import time
start = time.time()
# ... your code ...
print(f"Took {time.time() - start} seconds")
```

### Issue: "Memory usage increases over time"
**Symptoms**: Application gets slower, eventual crash

**Solutions**:
```bash
# 1. Clear Streamlit cache
streamlit cache clear

# 2. Reduce model size
# Load only necessary models, unload when done

# 3. Monitor memory
watch -n 1 'ps aux | grep python'

# 4. In code, be mindful of memory
del large_dataframe  # Free memory
gc.collect()  # Force garbage collection
```

### Issue: "Prediction takes >5 seconds"
**Symptoms**: Slow ML model inference

**Solutions**:
```python
# 1. Use simpler model (Logistic Regression instead of Random Forest)
# 2. Reduce feature size
# 3. Preprocess data more efficiently
# 4. Use model quantization
# 5. Consider GPU acceleration for neural networks

import tensorflow as tf
# Use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(physical_devices)}")
```

---

## Database Issues

### Issue: "sqlite3.OperationalError: database is locked"
**Symptoms**: Database lock when accessing users.db

**Solutions**:
```bash
# 1. Close all connections to database
# Stop all running instances of the application

# 2. Remove lock file
rm .Data/users.db-wal  # macOS/Linux
del Data\users.db-wal  # Windows

# 3. Restart application
streamlit run app.py

# 4. If persistent, reset database
rm Data/users.db
# Let app recreate it on next run
```

### Issue: "sqlite3.OperationalError: attempt to write a readonly database"
**Symptoms**: Cannot write to database

**Solutions**:
```bash
# 1. Check file permissions
ls -la Data/users.db  # macOS/Linux
icacls Data\users.db  # Windows

# 2. Grant write permissions
chmod 644 Data/users.db  # macOS/Linux

# 3. For Windows, use GUI:
# Right-click > Properties > Security > Edit > Select user > Full Control

# 4. Check disk space
df -h  # macOS/Linux
dir C:  # Windows (check available space)
```

### Issue: "Database corruption"
**Symptoms**: Unexpected database errors, data loss

**Solutions**:
```bash
# 1. Backup corrupted database
cp Data/users.db Data/users.db.corrupted

# 2. Restore from backup
cp Data/users.db.backup Data/users.db

# 3. If no backup, reset
rm Data/users.db
# Recreate on next application run

# 4. Check database integrity
sqlite3 Data/users.db "PRAGMA integrity_check;"
```

---

## Authentication Issues

### Issue: "Invalid token / Token expired"
**Symptoms**: Login fails with JWT error

**Solutions**:
```python
# 1. Clear session
st.session_state.clear()

# 2. Check JWT secret (must be same on all instances)
# In auth.py, verify secret_key is consistent

# 3. Logout and login again
# Navigate to login page and re-authenticate

# 4. Check server time synchronization
# JWT is time-sensitive
```

### Issue: "Password reset email not sent"
**Symptoms**: Email confirmation not working

**Solutions**:
```python
# 1. Check SMTP configuration
# Verify SMTP_SERVER and SMTP_PORT in .env

# 2. Enable "Less secure app access" (Gmail)
# https://myaccount.google.com/lesssecureapps

# 3. Use app password instead of account password
# https://myaccount.google.com/apppasswords

# 4. Check firewall/network settings
# Ensure port 587 is open for outgoing connections
```

### Issue: "User cannot login after signup"
**Symptoms**: Signup succeeds but login fails

**Solutions**:
```python
# 1. Check database has user record
import sqlite3
conn = sqlite3.connect('Data/users.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM users WHERE username='test'")
print(cursor.fetchone())
conn.close()

# 2. Verify password hashing is consistent
# Ensure same hash function used in signup and login

# 3. Check user activation status
cursor.execute("SELECT is_active FROM users WHERE username='test'")

# 4. Reset user and try again
cursor.execute("DELETE FROM users WHERE username='test'")
conn.commit()
```

---

## ML Model Issues

### Issue: "Model prediction accuracy is low"
**Symptoms**: Predictions don't match expected diseases

**Solutions**:
```python
# 1. Check training data quality
# Review dataset.csv for outliers and missing values

# 2. Retrain model with more data
# Add more training examples

# 3. Tune hyperparameters
from sklearn.model_selection import GridSearchCV
# Perform hyperparameter tuning

# 4. Use ensemble of models
# Combine predictions from multiple models

# 5. Check feature preprocessing
# Verify features are properly scaled and encoded
```

### Issue: "Model file not found / cannot load model"
**Symptoms**: joblib.load() fails

**Solutions**:
```bash
# 1. Check model file exists
ls models/  # macOS/Linux
dir models  # Windows

# 2. Retrain and save model
python scripts/train_model.py

# 3. Check file is not corrupted
file models/disease_model.pkl  # macOS/Linux

# 4. Verify joblib version compatibility
pip show joblib
```

### Issue: "Memory error when training model"
**Symptoms**: Out of memory during model training

**Solutions**:
```python
# 1. Use smaller dataset
data = data.sample(n=5000)  # Use sample for testing

# 2. Use batch training
for batch in get_batches(data):
    model.fit(batch)

# 3. Switch to model that uses less memory
# Use SGDClassifier instead of RandomForest

# 4. Increase available memory
# Close other applications
# Use server with more RAM
```

---

## Data Issues

### Issue: "CSV file encoding error"
**Symptoms**: UnicodeDecodeError when loading CSV

**Solutions**:
```python
# Specify encoding explicitly
df = pd.read_csv('Data/dataset.csv', encoding='utf-8')

# Try alternative encodings
try:
    df = pd.read_csv('file.csv', encoding='latin-1')
except:
    df = pd.read_csv('file.csv', encoding='iso-8859-1')

# Check file encoding
file -i Data/dataset.csv  # macOS/Linux
```

### Issue: "Missing values in dataset"
**Symptoms**: NaN values causing errors

**Solutions**:
```python
# Handle missing values
df.fillna(df.mean(), inplace=True)  # Fill with mean
df.fillna(method='ffill', inplace=True)  # Forward fill
df.dropna(inplace=True)  # Remove rows with NaN

# Check for missing values
print(df.isnull().sum())
```

---

## Getting Help

If you don't find a solution here:

1. **Check GitHub Issues**: Search existing issues
2. **Create New Issue**: Provide error message, steps to reproduce
3. **Contact Support**: Email support@healthai.com
4. **Check Documentation**: Review README.md and other docs
5. **Ask Community**: GitHub Discussions

---

**Last Updated**: January 2026  
**Version**: 1.0.0
