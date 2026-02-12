# Quick Start Guide

Get up and running with Healthcare AI Recommendation System in 5 minutes!

## Prerequisites
- Python 3.8+
- Git
- 2GB free disk space

---

## Installation (Windows/macOS/Linux)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR-USERNAME/medi_recomm.git
cd medi_recomm
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run app.py
```

Application opens at: **http://localhost:8501**

---

## First Time Setup

### 1. Create Account
- Click "Sign Up"
- Enter username, email, password
- Add your health information
- Click "Create Account"

### 2. Login
- Click "Login"
- Enter credentials
- Click "Login"

### 3. Explore Features
- **Dashboard**: View health overview
- **Disease Prediction**: Enter symptoms, get predictions
- **Medicine Recommendation**: Get personalized medicine suggestions
- **Analytics**: View health trends
- **Profile**: Update your information

---

## Demo Credentials

For testing without signup (if admin creates demo account):
```
Username: demo
Password: demo@123
```

---

## Common Tasks

### Disease Prediction
1. Navigate to "🤒 Disease Prediction"
2. Select symptoms from dropdown
3. Enter your age and gender
4. Click "Predict"
5. View results and recommendations

### Get Medicine Recommendation
1. Navigate to "💊 Medicine Recommendation"
2. Select a disease or enter symptoms
3. View recommended medicines
4. Filter by rating or availability
5. Save favorites

### View Analytics
1. Navigate to "📊 Analytics"
2. Select date range (optional)
3. Explore different visualizations
4. Download report (PDF)

### Update Profile
1. Navigate to "👤 Profile"
2. Update health metrics (weight, height, age)
3. Add medical history
4. Save allergies and preferences
5. Click "Save Changes"

---

## Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Database Error
```bash
# Clear database and recreate
rm Data/users.db
streamlit run app.py
```

### Missing Data Files
```bash
# Ensure Data directory exists
mkdir Data

# Copy CSV files to Data directory
# - dataset.csv (required)
# - Drug.csv (required)
```

### Module Not Found
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

---

## Project Structure

```
medi_recomm/
├── app.py                    # Main application
├── auth.py                   # Authentication
├── models.py                 # ML Models
├── utils.py                  # Utilities
├── Data/                     # Data files
├── pages/                    # Application pages
└── README.md                 # Documentation
```

---

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main application entry |
| `auth.py` | User authentication |
| `models.py` | ML/DL models |
| `utils.py` | Data utilities |
| `Data/dataset.csv` | Disease-symptom data |
| `Data/Drug.csv` | Medicine database |

---

## Useful Commands

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Clear Streamlit cache
streamlit cache clear

# Run with debugging
streamlit run app.py --logger.level=debug

# Check Python version
python --version

# List installed packages
pip list

# Deactivate virtual environment
deactivate
```

---

## Next Steps

1. ✅ Installation complete!
2. 📖 Read [README.md](README.md) for full documentation
3. 🔧 Check [DEVELOPMENT.md](DEVELOPMENT.md) for development setup
4. 🤝 See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
5. 📊 Explore features and provide feedback

---

## Additional Resources

- 📚 [Full Documentation](README.md)
- 🛠️ [Development Guide](DEVELOPMENT.md)
- 🤝 [Contributing Guidelines](CONTRIBUTING.md)
- 🏗️ [Architecture Overview](ARCHITECTURE.md)
- 🐛 [Troubleshooting](TROUBLESHOOTING.md)
- 🗺️ [Project Roadmap](ROADMAP.md)

---

## Getting Help

### Need Help?
- 📧 Email: support@healthai.com
- 💬 GitHub Issues: [Create issue](../../issues)
- 🌐 GitHub Discussions: [Ask question](../../discussions)

### Report a Bug
- Describe the issue
- Steps to reproduce
- Expected vs actual result
- System information

### Suggest a Feature
- Describe your idea
- Explain the use case
- Provide examples

---

## Important Notes

⚠️ **Disclaimer**: This application is for educational purposes only. It is NOT a substitute for professional medical advice. Always consult qualified healthcare professionals.

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Status**: ✅ Ready to Use
