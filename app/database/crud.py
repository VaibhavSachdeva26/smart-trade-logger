# app/database/crud.py

from sqlalchemy.orm import Session
from app.database import models, schemas


def create_trade_log(db: Session, trade: schemas.TradeLogCreate) -> models.TradeLog:
    db_trade = models.TradeLog(**trade.dict())
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    return db_trade


# ---------------- Strategy Mappings ----------------
def add_strategy_mapping(db: Session, mapping: schemas.StrategyMappingSchema):
    db_item = models.StrategyMapping(**mapping.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_strategies_for_setup(db: Session, sentiment: str, setup_name: str):
    return db.query(models.StrategyMapping).filter_by(sentiment=sentiment, setup_name=setup_name).all()

# ---------------- Setup Metadata ----------------
def add_setup_metadata(db: Session, setup: schemas.SetupMetadataSchema):
    db_item = models.SetupMetadata(**setup.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_all_setups(db: Session):
    return db.query(models.SetupMetadata).all()

# ---------------- Tag Metadata ----------------
def add_tag(db: Session, tag: schemas.TagMetadataSchema):
    db_item = models.TagMetadata(**tag.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

# ---------------- User Behavior Logs ----------------
def add_behavior_log(db: Session, log: schemas.UserBehaviorLogSchema):
    db_item = models.UserBehaviorLog(**log.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

# ---------------- Uploaded Images ----------------
def add_uploaded_image(db: Session, image: schemas.UploadedImageSchema):
    db_item = models.UploadedImage(**image.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


from app.database.models import TradeHeader, TradeLeg
from sqlalchemy.orm import Session

# ---------------------------------------------------------
# ✅ Update exit price for equity or F&O legs
# ---------------------------------------------------------
def update_exit_prices(db: Session, trade_id: str, exit_data: dict, classification: str = None, remarks: str = None):
    """
    Update exit prices for a trade (both equity and F&O).
    exit_data example (for F&O): {leg_id: exit_price, leg_id2: exit_price2}
    exit_data example (for equity): {'equity_exit': 435.0}
    """
    trade = db.query(TradeHeader).filter(TradeHeader.id == trade_id).first()
    if not trade:
        return False, "Trade not found"

    # Update F&O legs if applicable
    if trade.strategy.lower() != "equity":
        legs = db.query(TradeLeg).filter(TradeLeg.trade_id == trade_id).all()
        for leg in legs:
            if leg.id in exit_data:
                leg.exit_price = exit_data[leg.id]
        db.flush()
    else:
        # For equity trades, store exit price directly on TradeHeader
        if "equity_exit" in exit_data:
            trade.exit_price = exit_data["equity_exit"]

    # Update classification / remarks
    if classification:
        trade.classification = classification
    if remarks:
        trade.notes = (trade.notes or "") + f"\n{remarks}"

    trade.is_closed = 1
    db.commit()
    db.refresh(trade)
    return True, "Exit prices updated successfully"
