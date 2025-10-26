# app/database/create_tables.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # app/frontend → smart_trade_logger → [parent]

from app.database.models import Base
from app.database.db import engine

# Creates all tables defined in models.py
print("⏳ Creating tables...")
Base.metadata.create_all(bind=engine)
print("✅ Tables created successfully.")
