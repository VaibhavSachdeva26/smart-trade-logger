# app/database/models.py
from sqlalchemy import Column, String, Float, Integer, Date, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import date

Base = declarative_base()

class TradeLog(Base):
    __tablename__ = "trade_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_date = Column(Date, nullable=False)
    sentiment = Column(String, nullable=False)
    setup = Column(String, nullable=False)
    strategy = Column(String, nullable=False)

    strike_prices = Column(String)
    premium_buy = Column(Float)
    premium_sell = Column(Float)
    stop_loss = Column(Float)
    target = Column(Float)

    capital_used = Column(Float)
    margin_required = Column(Float)

    emotion = Column(String)
    confidence = Column(Integer)
    classification = Column(String)
    notes = Column(Text)
    image_path = Column(String)

    exit_price = Column(Float)
    is_closed = Column(Integer, default=0)  # 0 = open, 1 = closed


from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

# --- Strategy Mapping ---
class StrategyMapping(Base):
    __tablename__ = "strategy_mappings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sentiment = Column(String, nullable=False)
    setup_name = Column(String, nullable=False)
    strategy_name = Column(String, nullable=False)

# --- Setup Metadata ---
class SetupMetadata(Base):
    __tablename__ = "setup_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)
    type = Column(String)  # e.g., "Bullish Swing", "Bearish Momentum"
    description = Column(Text)

# --- Tag Metadata ---
class TagMetadata(Base):
    __tablename__ = "tag_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tag_name = Column(String, nullable=False, unique=True)
    tag_description = Column(Text)

# --- User Behavior Logs ---
class UserBehaviorLog(Base):
    __tablename__ = "user_behavior_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trade_logs.id"))
    emotion = Column(String)
    confidence = Column(Integer)
    notes = Column(Text)
    timestamp = Column(Date, default=date.today)

# --- Uploaded Images ---
class UploadedImage(Base):
    __tablename__ = "uploaded_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trade_logs.id"))
    image_path = Column(String)
    description = Column(Text)
    uploaded_at = Column(Date, default=date.today)

# app/database/models.py

class TradeHeader(Base):
    __tablename__ = "trade_headers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_date = Column(Date, nullable=False)
    symbol = Column(String, nullable=False)
    wave_timeframe = Column(String, nullable=True)
    sentiment = Column(String)
    setup = Column(String)
    strategy = Column(String)

    # 🔽 ADD THESE FOUR LINES
    entry_price = Column(Float)        # equity entry (or spot at entry)
    stop_loss = Column(Float)          # equity SL
    target = Column(Float)             # equity target
    exit_price = Column(Float)         # equity exit (optional — we also store in leg for consistency)

    capital_used = Column(Float)
    margin_required = Column(Float)
    lot_size = Column(Integer)
    num_lots = Column(Integer)
    emotion = Column(String)
    confidence = Column(Integer)
    classification = Column(String)
    notes = Column(Text)
    image_path = Column(String)
    is_closed = Column(Integer, default=0)

    legs = relationship("TradeLeg", back_populates="trade", cascade="all, delete-orphan")

class TradeLeg(Base):
    __tablename__ = "trade_legs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trade_headers.id"))
    label = Column(String)
    action = Column(String)
    option_type = Column(String)
    strike = Column(Float)
    premium = Column(Float)
    exit_price = Column(Float)

    trade = relationship("TradeHeader", back_populates="legs")