# app/database/create_tables.py

from app.database.db import engine, SessionLocal
from app.database.models import Base, User
import hashlib

def create_default_admin():
    """Create default admin user with hashed password if not exists."""
    db = SessionLocal()
    username = "VaibhavSachdeva"
    raw_password = "VaibhavSachdeva@026"

    hashed_password = hashlib.sha256(raw_password.encode()).hexdigest()

    existing_user = db.query(User).filter_by(username=username).first()
    if not existing_user:
        new_user = User(username=username, password=hashed_password, is_admin=True)
        db.add(new_user)
        db.commit()
        print(f"✅ Default user '{username}' created successfully (SHA-256 hashed).")
    else:
        print(f"ℹ️ User '{username}' already exists.")
    db.close()

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully.")
    create_default_admin()
