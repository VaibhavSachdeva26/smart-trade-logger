# app/database/schemas.py

from pydantic import BaseModel
from typing import Optional
from datetime import date
import uuid


class TradeLogBase(BaseModel):
    trade_date: date
    sentiment: str
    setup: str
    strategy: str
    strike_prices: Optional[str] = None
    premium_buy: Optional[float] = 0.0
    premium_sell: Optional[float] = 0.0
    stop_loss: Optional[float] = 0.0
    target: Optional[float] = 0.0
    wave_timeframe: Optional[str] = None
    capital_used: Optional[float] = 0.0
    margin_required: Optional[float] = 0.0
    emotion: Optional[str] = None
    confidence: Optional[int] = None
    classification: Optional[str] = None
    notes: Optional[str] = None
    image_path: Optional[str] = None


class TradeLogCreate(TradeLogBase):
    pass


class TradeLogInDB(TradeLogBase):
    id: uuid.UUID

    class Config:
        orm_mode = True

from pydantic import BaseModel
from typing import Optional
from datetime import date
from uuid import UUID

# -------------------- Strategy Mapping --------------------
class StrategyMappingSchema(BaseModel):
    sentiment: str
    setup_name: str
    strategy_name: str

# -------------------- Setup Metadata --------------------
class SetupMetadataSchema(BaseModel):
    name: str
    type: Optional[str]
    description: Optional[str]

# -------------------- Tag Metadata --------------------
class TagMetadataSchema(BaseModel):
    tag_name: str
    tag_description: Optional[str]

# -------------------- User Behavior Log --------------------
class UserBehaviorLogSchema(BaseModel):
    trade_id: UUID
    emotion: Optional[str]
    confidence: Optional[int]
    notes: Optional[str]
    timestamp: Optional[date]

# -------------------- Uploaded Image --------------------
class UploadedImageSchema(BaseModel):
    trade_id: UUID
    image_path: str
    description: Optional[str]
    uploaded_at: Optional[date]
