import streamlit as st
import pandas as pd
import hashlib
import sqlite3
import re
from datetime import datetime, timedelta
import jwt
import json
from typing import Optional, Dict, Tuple
import secrets
import string

class UserAuthentication:
    """Complete user authentication and management system"""
    
    def __init__(self, db_path='Data/users.db'):
        self.db_path = db_path
        self.secret_key = 'healthcare_ai_secret_key_2024'  # In production, use environment variable
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for user management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                age INTEGER,
                gender TEXT,
                blood_group TEXT,
                allergies TEXT,
                medical_history TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        
        # User activity logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                activity_type TEXT,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User preferences
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE,
                theme TEXT DEFAULT 'dark',
                language TEXT DEFAULT 'english',
                notifications_enabled INTEGER DEFAULT 1,
                email_notifications INTEGER DEFAULT 1,
                dashboard_layout TEXT,
                medication_reminders INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User medical data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_medical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                height_cm REAL,
                weight_kg REAL,
                blood_pressure TEXT,
                glucose_level REAL,
                cholesterol TEXT,
                bmi REAL,
                symptoms TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = 'healthcare_salt_2024'
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one number"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is strong"
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def generate_token(self, user_id: int, username: str) -> str:
        """Generate JWT token for authentication"""
        payload = {
            'user_id': user_id,
            'username': username,
            'exp': datetime.utcnow() + timedelta(days=7)
        }
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        # PyJWT 2.0+ returns bytes, so decode to string
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            # Ensure token is string
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def register_user(self, user_data: Dict) -> Tuple[bool, str]:
        """Register new user"""
        try:
            # Validate email
            if not self.validate_email(user_data['email']):
                return False, "Invalid email format"
            
            # Validate password strength
            is_valid, message = self.validate_password_strength(user_data['password'])
            if not is_valid:
                return False, message
            
            # Check if user already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', 
                          (user_data['username'], user_data['email']))
            if cursor.fetchone():
                conn.close()
                return False, "Username or email already exists"
            
            # Hash password
            password_hash = self.hash_password(user_data['password'])
            
            # Insert new user
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, age, gender, 
                                 blood_group, allergies, medical_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_data['username'],
                user_data['email'],
                password_hash,
                user_data.get('full_name', ''),
                user_data.get('age'),
                user_data.get('gender', ''),
                user_data.get('blood_group', ''),
                json.dumps(user_data.get('allergies', [])),
                json.dumps(user_data.get('medical_history', []))
            ))
            
            user_id = cursor.lastrowid
            
            # Create default preferences
            cursor.execute('''
                INSERT INTO user_preferences (user_id) VALUES (?)
            ''', (user_id,))
            
            # Log registration activity
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details)
                VALUES (?, ?, ?)
            ''', (user_id, 'REGISTRATION', 'New user registered'))
            
            conn.commit()
            conn.close()
            
            return True, "Registration successful"
            
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Authenticate user and return token"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, password_hash, is_active, is_admin 
                FROM users 
                WHERE (username = ? OR email = ?) AND is_active = 1
            ''', (username, username))
            
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return False, "Invalid username or password", None
            
            user_id, db_username, password_hash, is_active, is_admin = user
            
            # Verify password
            if self.hash_password(password) != password_hash:
                conn.close()
                return False, "Invalid username or password", None
            
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            ''', (user_id,))
            
            # Log login activity
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details)
                VALUES (?, ?, ?)
            ''', (user_id, 'LOGIN', 'User logged in'))
            
            # Get user profile
            cursor.execute('''
                SELECT username, email, full_name, age, gender, blood_group, 
                       allergies, medical_history, created_at
                FROM users WHERE id = ?
            ''', (user_id,))
            
            user_info = cursor.fetchone()
            
            user_profile = {
                'id': user_id,
                'username': user_info[0],
                'email': user_info[1],
                'full_name': user_info[2],
                'age': user_info[3],
                'gender': user_info[4],
                'blood_group': user_info[5],
                'allergies': json.loads(user_info[6]) if user_info[6] else [],
                'medical_history': json.loads(user_info[7]) if user_info[7] else [],
                'created_at': user_info[8],
                'is_admin': bool(is_admin)
            }
            
            conn.commit()
            conn.close()
            
            # Generate token
            token = self.generate_token(user_id, db_username)
            
            return True, "Authentication successful", {'token': token, 'user': user_profile}
            
        except Exception as e:
            return False, f"Authentication error: {str(e)}", None
    
    def update_user_profile(self, user_id: int, profile_data: Dict) -> Tuple[bool, str]:
        """Update user profile"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = []
            update_values = []
            
            allowed_fields = ['full_name', 'age', 'gender', 'blood_group', 'allergies', 'medical_history']
            
            for field in allowed_fields:
                if field in profile_data:
                    if field in ['allergies', 'medical_history']:
                        update_values.append(json.dumps(profile_data[field]))
                    else:
                        update_values.append(profile_data[field])
                    update_fields.append(f"{field} = ?")
            
            if update_fields:
                update_values.append(user_id)
                query = f'''
                    UPDATE users 
                    SET {', '.join(update_fields)}
                    WHERE id = ?
                '''
                cursor.execute(query, update_values)
                
                # Log profile update
                cursor.execute('''
                    INSERT INTO user_activity (user_id, activity_type, details)
                    VALUES (?, ?, ?)
                ''', (user_id, 'PROFILE_UPDATE', 'User updated profile'))
                
                conn.commit()
            
            conn.close()
            return True, "Profile updated successfully"
            
        except Exception as e:
            return False, f"Profile update failed: {str(e)}"
    
    def update_user_preferences(self, user_id: int, preferences: Dict) -> Tuple[bool, str]:
        """Update user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = []
            update_values = []
            
            allowed_fields = ['theme', 'language', 'notifications_enabled', 
                            'email_notifications', 'dashboard_layout', 'medication_reminders']
            
            for field in allowed_fields:
                if field in preferences:
                    update_values.append(preferences[field])
                    update_fields.append(f"{field} = ?")
            
            if update_fields:
                update_values.append(user_id)
                
                # Check if preferences exist
                cursor.execute('SELECT id FROM user_preferences WHERE user_id = ?', (user_id,))
                if cursor.fetchone():
                    query = f'''
                        UPDATE user_preferences 
                        SET {', '.join(update_fields)}
                        WHERE user_id = ?
                    '''
                else:
                    update_fields.append('user_id')
                    update_values.insert(len(update_fields)-1, user_id)
                    query = f'''
                        INSERT INTO user_preferences ({', '.join(update_fields)})
                        VALUES ({', '.join(['?' for _ in update_fields])})
                    '''
                
                cursor.execute(query, update_values)
                conn.commit()
            
            conn.close()
            return True, "Preferences updated successfully"
            
        except Exception as e:
            return False, f"Preferences update failed: {str(e)}"
    
    def add_medical_data(self, user_id: int, medical_data: Dict) -> Tuple[bool, str]:
        """Add user medical data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_medical_data 
                (user_id, height_cm, weight_kg, blood_pressure, glucose_level, 
                 cholesterol, bmi, symptoms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                medical_data.get('height_cm'),
                medical_data.get('weight_kg'),
                medical_data.get('blood_pressure'),
                medical_data.get('glucose_level'),
                medical_data.get('cholesterol'),
                medical_data.get('bmi'),
                json.dumps(medical_data.get('symptoms', []))
            ))
            
            # Log medical data update
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details)
                VALUES (?, ?, ?)
            ''', (user_id, 'MEDICAL_DATA_UPDATE', 'User added medical data'))
            
            conn.commit()
            conn.close()
            
            return True, "Medical data added successfully"
            
        except Exception as e:
            return False, f"Medical data addition failed: {str(e)}"
    
    def get_user_medical_history(self, user_id: int, limit: int = 10) -> pd.DataFrame:
        """Get user medical history"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT recorded_at, height_cm, weight_kg, blood_pressure, 
                       glucose_level, cholesterol, bmi, symptoms
                FROM user_medical_data 
                WHERE user_id = ?
                ORDER BY recorded_at DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(user_id, limit))
            conn.close()
            
            # Parse symptoms from JSON
            if 'symptoms' in df.columns:
                df['symptoms'] = df['symptoms'].apply(lambda x: ', '.join(json.loads(x)) if x else '')
            
            return df
            
        except Exception as e:
            print(f"Error getting medical history: {e}")
            return pd.DataFrame()
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current password hash
            cursor.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False, "User not found"
            
            current_hash = result[0]
            
            # Verify old password
            if self.hash_password(old_password) != current_hash:
                conn.close()
                return False, "Current password is incorrect"
            
            # Validate new password strength
            is_valid, message = self.validate_password_strength(new_password)
            if not is_valid:
                conn.close()
                return False, message
            
            # Update password
            new_hash = self.hash_password(new_password)
            cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user_id))
            
            # Log password change
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details)
                VALUES (?, ?, ?)
            ''', (user_id, 'PASSWORD_CHANGE', 'User changed password'))
            
            conn.commit()
            conn.close()
            
            return True, "Password changed successfully"
            
        except Exception as e:
            return False, f"Password change failed: {str(e)}"
    
    def reset_password_request(self, email: str) -> Tuple[bool, str, Optional[str]]:
        """Request password reset (generates reset token)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, username FROM users WHERE email = ? AND is_active = 1', (email,))
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return False, "Email not found", None
            
            user_id, username = user
            
            # Generate reset token (simple for demo, use proper token in production)
            reset_token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
            
            # Store reset token (in production, store in database with expiration)
            # For demo, we'll just return it
            
            # Log reset request
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details)
                VALUES (?, ?, ?)
            ''', (user_id, 'PASSWORD_RESET_REQUEST', 'User requested password reset'))
            
            conn.commit()
            conn.close()
            
            return True, "Password reset email sent", reset_token
            
        except Exception as e:
            return False, f"Password reset request failed: {str(e)}", None
    
    def reset_password(self, token: str, new_password: str) -> Tuple[bool, str]:
        """Reset password using token"""
        try:
            # In production, verify token from database
            # For demo, we'll skip token verification
            
            # Validate password strength
            is_valid, message = self.validate_password_strength(new_password)
            if not is_valid:
                return False, message
            
            return True, "Password reset successful"
            
        except Exception as e:
            return False, f"Password reset failed: {str(e)}"
    
    def get_user_statistics(self) -> Dict:
        """Get system user statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total users
            cursor.execute('SELECT COUNT(*) FROM users')
            total_users = cursor.fetchone()[0]
            
            # Active users
            cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
            active_users = cursor.fetchone()[0]
            
            # Today's logins
            cursor.execute('''
                SELECT COUNT(DISTINCT user_id) 
                FROM user_activity 
                WHERE activity_type = 'LOGIN' 
                AND DATE(timestamp) = DATE('now')
            ''')
            today_logins = cursor.fetchone()[0]
            
            # New users this week
            cursor.execute('''
                SELECT COUNT(*) 
                FROM users 
                WHERE DATE(created_at) >= DATE('now', '-7 days')
            ''')
            new_users_week = cursor.fetchone()[0]
            
            # Gender distribution
            cursor.execute('''
                SELECT gender, COUNT(*) 
                FROM users 
                WHERE gender IS NOT NULL AND gender != ''
                GROUP BY gender
            ''')
            gender_dist = dict(cursor.fetchall())
            
            # Age distribution
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN age < 18 THEN 'Under 18'
                        WHEN age BETWEEN 18 AND 30 THEN '18-30'
                        WHEN age BETWEEN 31 AND 50 THEN '31-50'
                        WHEN age > 50 THEN 'Over 50'
                        ELSE 'Not specified'
                    END as age_group,
                    COUNT(*)
                FROM users
                GROUP BY age_group
            ''')
            age_dist = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'today_logins': today_logins,
                'new_users_week': new_users_week,
                'gender_distribution': gender_dist,
                'age_distribution': age_dist
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def get_recent_activity(self, user_id: Optional[int] = None, limit: int = 10) -> pd.DataFrame:
        """Get recent user activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if user_id:
                query = '''
                    SELECT ua.timestamp, ua.activity_type, ua.details, u.username
                    FROM user_activity ua
                    JOIN users u ON ua.user_id = u.id
                    WHERE ua.user_id = ?
                    ORDER BY ua.timestamp DESC
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, conn, params=(user_id, limit))
            else:
                query = '''
                    SELECT ua.timestamp, ua.activity_type, ua.details, u.username
                    FROM user_activity ua
                    JOIN users u ON ua.user_id = u.id
                    ORDER BY ua.timestamp DESC
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, conn, params=(limit,))
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"Error getting activity: {e}")
            return pd.DataFrame()

# Singleton instance
auth_system = UserAuthentication()

def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'

def check_authentication():
    """Check if user is authenticated"""
    if st.session_state.authenticated and st.session_state.token:
        try:
            # Verify token
            payload = auth_system.verify_token(st.session_state.token)
            if payload:
                return True
        except:
            pass
    
    # Clear session if not authenticated
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.token = None
    return False

def logout():
    """Logout user"""
    if st.session_state.user:
        # Log logout activity
        try:
            conn = sqlite3.connect(auth_system.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details)
                VALUES (?, ?, ?)
            ''', (st.session_state.user['id'], 'LOGOUT', 'User logged out'))
            conn.commit()
            conn.close()
        except:
            pass
    
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.token = None
    st.session_state.current_page = 'login'
    st.rerun()

# Singleton instance
auth_system = UserAuthentication()