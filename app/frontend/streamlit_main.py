import streamlit as st
from datetime import date, datetime, timedelta
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import get_script_run_ctx
import extra_streamlit_components as stx
from datetime import datetime, timedelta
import time

cookie_manager = stx.CookieManager(key="smarttrade_cookie_manager")

from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
from io import BytesIO
import json
import hashlib
import streamlit.components.v1 as components

import re
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.database.db import SessionLocal
from app.database.models import TradeHeader, TradeLeg, User
from app.utils.strategy_mapping import (
    get_sentiment_types,
    get_setups_by_sentiment,
    get_strategies_by_sentiment,
)
from app.utils.strategy_leg_config import STRATEGY_LEG_CONFIG
from PIL import Image
import io
import extra_streamlit_components as stx

# Create ONE global cookie manager with a fixed unique key
cookie_manager = stx.CookieManager(key="login_cookie_manager")

# --- Helper to estimate option premium at new spot using delta ---
def _estimate_exit_premium(entry_prem: float, delta: float, option_type: str,
                           spot_entry: float, spot_exit: float) -> float:
    """
    Estimate option premium at spot_exit using instrument delta (CALL +, PUT -).
    Normalizes delta sign by option type so user can enter +/− freely.
    """
    opt = (option_type or "").upper()
    if opt == "CALL":
        eff_delta = abs(delta)
    elif opt == "PUT":
        eff_delta = -abs(delta)
    else:
        eff_delta = 0.0
    est = entry_prem + eff_delta * (spot_exit - spot_entry)
    return max(0.0, est)

# ======================================================
# LOGIN + PERSISTENT SESSION HANDLER
# ======================================================
import extra_streamlit_components as stx
from datetime import datetime, timedelta
import hashlib
import time

cookie_manager = stx.CookieManager(key="smarttrade_cookie_mgr")

# -----------------------------
# LOGIN HELPER FUNCTION
# -----------------------------
def login():
    """Render login form and handle authentication"""
    st.title("🔐 Smart Trade Logger – Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()

        db = SessionLocal()
        user = db.query(User).filter_by(username=username, password=hashed_pw).first()
        db.close()

        if user:
            expiry = datetime.now() + timedelta(days=7)

            # ✅ Set cookies to persist login
            cookie_manager.set("logged_in", "true", key="login_cookie", expires_at=expiry)
            cookie_manager.set("username", username, key="user_cookie", expires_at=expiry)

            # ✅ Persist session state
            st.session_state["logged_in"] = True
            st.session_state["username"] = username

            st.success("✅ Login successful! Reloading...")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("❌ Invalid username or password")


# -----------------------------
# AUTO LOGIN HANDLER
# -----------------------------
def ensure_login():
    """Restore login from cookie or show login page"""
    # Wait for cookie manager to load
    time.sleep(0.3)
    cookies = cookie_manager.get_all(key="load_cookie")

    # ✅ Restore from cookies
    if cookies.get("logged_in") == "true" and cookies.get("username"):
        st.session_state["logged_in"] = True
        st.session_state["username"] = cookies.get("username")

    # ✅ Check session state
    if not st.session_state.get("logged_in", False):
        login()
        st.stop()


# ======================================================
# CALL THIS BEFORE SIDEBAR / MAIN PAGES
# ======================================================
ensure_login()

# ✅ Add logout button in sidebar
if st.sidebar.button("🚪 Logout"):
    cookie_manager.delete("logged_in", key="logout_cookie")
    cookie_manager.delete("username", key="logout_user_cookie")
    st.session_state.clear()
    st.success("✅ Logged out successfully!")
    st.rerun()

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["📥 Log Trade", "📋 View Trade Logs", "📈 Exit / Update Trades", "📜 Code of Conduct"],
)


def _legs_to_human(legs):
    parts = []
    for l in legs:
        parts.append(f"{l.label}@{l.strike}")
    return ", ".join(parts)

def calculate_spread_metrics(strategy_type, legs, lot_size, num_lots, stop_loss_spot=None, entry_spot=None):
    if not legs:
        return None

    lot_qty = (lot_size or 1) * (num_lots or 1)
    strategy_type_lower = strategy_type.lower()

    # Extract leg data by type
    calls = [l for l in legs if l.option_type.upper() == "CALL"]
    puts = [l for l in legs if l.option_type.upper() == "PUT"]
    calls.sort(key=lambda x: x.strike)
    puts.sort(key=lambda x: x.strike)

    max_profit = max_loss = bep = None

    # ===========================
    # Two-leg Spreads
    # ===========================
    if len(legs) == 2:
        long_leg = next((l for l in legs if l.action.lower() == "buy"), None)
        short_leg = next((l for l in legs if l.action.lower() == "sell"), None)
        if not long_leg or not short_leg:
            return None

        strike_diff = abs(short_leg.strike - long_leg.strike)
        net_credit = (short_leg.premium - long_leg.premium)
        net_debit = -net_credit

        if "bull call" in strategy_type_lower or "bear put" in strategy_type_lower:
            # Debit spreads
            max_profit = (strike_diff - net_debit) * lot_qty
            max_loss = net_debit * lot_qty
            bep = long_leg.strike + net_debit if "bull" in strategy_type_lower else long_leg.strike - net_debit

        elif "bull put" in strategy_type_lower or "bear call" in strategy_type_lower:
            # Credit spreads
            max_profit = net_credit * lot_qty
            max_loss = (strike_diff - net_credit) * lot_qty
            bep = short_leg.strike - net_credit if "bull" in strategy_type_lower else short_leg.strike + net_credit

        elif "straddle" in strategy_type_lower or "strangle" in strategy_type_lower:
            ce = calls[0] if calls else None
            pe = puts[0] if puts else None
            if ce and pe:
                net_premium = ce.premium + pe.premium
                atm_strike = (ce.strike + pe.strike) / 2
                if "short" in strategy_type_lower:
                    max_profit = net_premium * lot_qty
                    max_loss = "Unlimited"
                    bep = f"{round(atm_strike - net_premium,2)} / {round(atm_strike + net_premium,2)}"
                else:
                    max_profit = "Unlimited"
                    max_loss = net_premium * lot_qty
                    bep = f"{round(atm_strike - net_premium,2)} / {round(atm_strike + net_premium,2)}"

    # ===========================
    # Iron Condor / Butterfly
    # ===========================
    elif len(legs) == 4 and ("iron condor" in strategy_type_lower or "iron butterfly" in strategy_type_lower):
        short_put = min([l for l in puts if l.action.lower() == "sell"], key=lambda x: x.strike, default=None)
        long_put = min([l for l in puts if l.action.lower() == "buy"], key=lambda x: x.strike, default=None)
        short_call = max([l for l in calls if l.action.lower() == "sell"], key=lambda x: x.strike, default=None)
        long_call = max([l for l in calls if l.action.lower() == "buy"], key=lambda x: x.strike, default=None)

        if not (short_put and long_put and short_call and long_call):
            return None

        credit_put = short_put.premium - long_put.premium
        credit_call = short_call.premium - long_call.premium
        total_credit = credit_put + credit_call
        width = short_call.strike - short_put.strike

        max_profit = total_credit * lot_qty
        max_loss = (width - total_credit) * lot_qty
        bep = f"{round(short_put.strike - credit_put,2)} / {round(short_call.strike + credit_call,2)}"

    # ===========================
    # Ladder or Ratio spreads
    # ===========================
    elif len(legs) == 3 and ("ratio" in strategy_type_lower or "ladder" in strategy_type_lower):
        net_premium = sum([l.premium * (1 if l.action.lower() == "sell" else -1) for l in legs])
        strikes = [l.strike for l in legs]
        strike_range = max(strikes) - min(strikes)
        max_profit = (strike_range - abs(net_premium)) * lot_qty
        max_loss = abs(net_premium) * lot_qty
        bep = f"{min(strikes)+abs(net_premium)} / {max(strikes)-abs(net_premium)}"

    # ===========================
    # Calculate Risk at Stop-Loss (Delta-based refined)
    # ===========================
    risk_at_sl = None
    if stop_loss_spot and entry_spot and legs:
        per_share_pnl = 0.0
        for leg in legs:
            entry = float(leg.premium or 0.0)
            d = float(leg.delta or 0.0)
            opt = (leg.option_type or "").upper()
            pos_sign = 1 if (leg.action or "").lower() == "buy" else -1

            exit_est = _estimate_exit_premium(entry, d, opt, entry_spot, stop_loss_spot)
            per_share_pnl += pos_sign * (exit_est - entry)

        risk_at_sl = round(per_share_pnl * lot_qty, 2)

    return {
        "bep": bep,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "risk_at_sl": risk_at_sl,
    }

# ------------------------------------------------------------
# PAGE 1: LOG TRADE
# ------------------------------------------------------------
if page == "📥 Log Trade":
    st.title("📘 Smart Trade Logger")
    st.subheader("Log and validate your stock market trades with full discipline")

    st.sidebar.header("Trade Type Selection")
    trade_type = st.sidebar.radio("Select Trade Type", ["Equity", "F&O"])
    trade_date = st.sidebar.date_input("Trade Date", value=date.today())

    if trade_type == "Equity":
        st.markdown("### 📈 Equity Trade Details")

        st.sidebar.header("Equity Configuration")

        sentiment = st.sidebar.selectbox("Market Sentiment", get_sentiment_types(), key="equity_sentiment")
        setups = get_setups_by_sentiment(sentiment)
        setup_name = st.sidebar.selectbox("Setup", setups, key="equity_setup")

        wave_timeframe = st.selectbox(
            "Wave / Timeframe",
            options=["5m", "15m", "1H", "4H", "1D", "1W", "1M"],
            index=4  # optional, sets "1D" as default
        )

        stock_symbol = st.text_input("Stock / Symbol")
        entry_price = st.number_input("Entry Price (₹)", min_value=0.0, format="%.2f")
        stop_loss = st.number_input("Stop Loss (₹)", min_value=0.0, format="%.2f")
        target = st.number_input("Target (₹)", min_value=0.0, format="%.2f")
        quantity = st.number_input("Quantity", min_value=0, step=1)

        capital_auto = entry_price * quantity
        st.markdown(f"💰 **Auto-Calculated Capital Used:** ₹{capital_auto:,.2f}")

        capital_used = st.number_input(
            "Override Capital Used (₹) — optional",
            min_value=0.0,
            value=capital_auto,
            key="equity_capital_used",
        )

        rr_ratio = None
        low_rr_warning = False
        if entry_price and stop_loss and target:
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            rr_ratio = round(reward / risk, 2) if risk else 0
            st.info(f"📊 Risk–Reward Ratio: {rr_ratio}:1")
            if rr_ratio < 3:
                st.error("⚠️ R:R < 1:3.")
                low_rr_warning = True

        uploaded_image = st.file_uploader("Upload or Paste Chart Screenshot", type=["png", "jpg", "jpeg"])
        emotion = st.select_slider("Emotion While Taking Trade", ["Calm", "Neutral", "Excited", "Fearful"])
        confidence = st.slider("Confidence (0-100)", 0, 100, 70)
        notes = st.text_area("Notes / Rationale")

        # 🧠 Prevent multiple rapid submissions, but reset automatically
        if "equity_submitted_once" not in st.session_state:
            st.session_state["equity_submitted_once"] = False

        if st.button("Submit Equity Trade"):
            st.session_state["equity_submitted_once"] = True

            required_equity_fields_filled = (
                    stock_symbol.strip() != ""
                    and entry_price > 0
                    and stop_loss > 0
                    and target > 0
                    and quantity > 0
            )

            if not required_equity_fields_filled:
                st.warning("⚠️ Please fill all required fields (symbol, prices, quantity, and capital).")
                st.stop()

            if low_rr_warning and not st.session_state.get("equity_rr_confirmed", False):
                st.warning("⚠️ Risk–Reward ratio below 1:3. Confirm to proceed.")
                st.session_state["equity_rr_confirmed"] = True
                st.stop()

            image_path = None
            if uploaded_image is not None:
                folder = Path("uploads") / str(trade_date)
                folder.mkdir(parents=True, exist_ok=True)

                filename = f"{stock_symbol.strip()}_{setup_name.strip()}_{wave_timeframe.strip()}_Equity.jpg".replace(" ", "_")
                image_path = folder / filename

                try:
                    img = Image.open(uploaded_image)
                    img = img.convert("RGB")  # Ensure JPEG-compatible
                    img.save(image_path, format="JPEG", optimize=True, quality=65)
                except Exception as e:
                    with open(image_path, "wb") as f:
                        f.write(uploaded_image.read())

            db = SessionLocal()
            try:
                header = TradeHeader(
                    trade_date=trade_date,
                    symbol=stock_symbol,
                    sentiment=sentiment,
                    setup=setup_name,
                    strategy="Equity",
                    capital_used=capital_used,
                    wave_timeframe=wave_timeframe,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target=target,
                    margin_required=0.0,
                    lot_size=quantity,  # treating qty as lot_size for equity is OK in your current schema
                    num_lots=1,
                    emotion=emotion,
                    confidence=confidence,
                    notes=notes,
                    image_path=str(image_path) if image_path else None,
                )

                db.add(header)
                db.commit()
                st.session_state["equity_submitted_once"] = False

                st.success("✅ Equity trade saved successfully!")
                if image_path:
                    st.image(str(image_path), caption="Uploaded Chart", width=True)
            except Exception as e:
                db.rollback()
                st.error(str(e))
            finally:
                db.close()

    else:
        st.markdown("### 🔁 F&O Strategy Trade Details")

        fno_symbol = st.text_input("Symbol / Scrip (e.g., NIFTY, BANKNIFTY)")
        wave_timeframe = st.selectbox(
            "Wave / Timeframe",
            options=["5m", "15m", "1H", "4H", "1D", "1W", "1M"],
            index=2  # optional, sets "1H" as default
        )

        margin_required = st.sidebar.number_input("Margin Required (₹)", min_value=0.0)


        sentiment = st.sidebar.selectbox("Market Sentiment", get_sentiment_types())
        setups = get_setups_by_sentiment(sentiment)
        setup_name = st.sidebar.selectbox("Setup", setups)
        strategies = get_strategies_by_sentiment(sentiment)
        strategy_type = st.sidebar.selectbox("F&O Strategy", strategies)
        # 🟩 Spot Chart Section (applies to all F&O strategies)
        st.markdown("### 📊 Spot Chart (for reference)")
        col_spot1, col_spot2, col_spot3 = st.columns(3)

        with col_spot1:
            spot_price = st.number_input("Spot Price (₹)", min_value=0.0, format="%.2f")
        with col_spot2:
            spot_sl = st.number_input("Spot Stop Loss (₹)", min_value=0.0, format="%.2f")
        with col_spot3:
            spot_target = st.number_input("Spot Target (₹)", min_value=0.0, format="%.2f")

        with st.form("fno_trade_form", clear_on_submit=False):

            legs = STRATEGY_LEG_CONFIG.get(strategy_type, [])
            leg_data = []
            total_premium = 0.0

            for i, leg in enumerate(legs):
                st.write(f"#### {leg['label']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    strike = st.number_input(f"{leg['label']} Strike", min_value=0.0, step=50.0, key=f"strike_{i}")
                with col2:
                    premium = st.number_input(f"{leg['label']} Premium", min_value=0.0, step=0.1, key=f"premium_{i}")
                with col3:
                    delta = st.number_input(f"{leg['label']} Δ", min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
                                            key=f"delta_{i}")

                leg_data.append({
                    "label": leg["label"],
                    "action": leg["action"],
                    "option_type": leg["option_type"],
                    "strike": strike,
                    "premium": premium,
                    "delta": delta,  # ✅ store leg-specific delta
                })

                total_premium += premium if leg["action"].lower() == "sell" else -premium

            colA, colB = st.columns(2)
            with colA:
                num_lots = st.number_input("Total Lots", min_value=1, step=1)
            with colB:
                lot_size = st.number_input("Lot Size", min_value=1, step=1)

            net_credit = total_premium * num_lots * lot_size
            st.success(f"💰 Net Premium ({'Credit' if net_credit>0 else 'Debit'}) ₹{abs(net_credit):,.2f}")

            capital_auto = margin_required + max(0, -net_credit)
            st.markdown(
                f"💰 **Auto-Calculated Capital Used:** ₹{abs(capital_auto):,.2f} "
                f"({'Credit Spread' if net_credit > 0 else 'Debit Spread'})"
            )
            capital_used = st.number_input(
                "Override Capital Used (₹) — optional",
                min_value=0.0,
                value=abs(capital_auto),
            )
            breakeven = None
            strikes = [l["strike"] for l in leg_data if l["strike"] > 0]
            if len(strikes) == 2 and "Spread" in strategy_type:
                if "Put" in strategy_type:
                    breakeven = strikes[0] - (net_credit / (num_lots * lot_size))
                elif "Call" in strategy_type:
                    breakeven = strikes[0] + (net_credit / (num_lots * lot_size))
            if breakeven:
                st.info(f"📈 Breakeven ≈ {round(breakeven,2)}")
            avg_delta = st.number_input(
                "Average Delta (e.g., 0.25 for Credit Spreads, 0.5 for Long Options)",
                min_value=-1.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
            )

            emotion = st.select_slider("Emotion While Taking Trade", ["Calm", "Neutral", "Excited", "Fearful"])
            confidence = st.slider("Confidence (0-100)", 0, 100, 70)
            notes = st.text_area("Notes / Rationale")
            uploaded_image = st.file_uploader("Upload or Paste Chart Screenshot", type=["png", "jpg", "jpeg"])

            required_fno_fields_filled = (
                    fno_symbol.strip() != ""
                    and margin_required > 0
                    and spot_price > 0
                    and spot_sl > 0
                    and spot_target > 0
                    and lot_size > 0
                    and num_lots > 0
                    and len(leg_data) > 0
                    and all(l["strike"] > 0 and l["premium"] > 0 for l in leg_data)
            )

            if not required_fno_fields_filled:
                st.warning("⚠️ Please fill all required fields before submitting.")

            submitted = st.form_submit_button("Submit F&O Trade", use_container_width=True)

            # 🧠 Prevent double submission (auto resets after success)
            if "fno_submitted_once" not in st.session_state:
                st.session_state["fno_submitted_once"] = False

            if submitted:
                if st.session_state["fno_submitted_once"]:
                    st.warning("⚠️ You already submitted this trade. Please reload the page to log a new one.")
                    st.stop()
                else:
                    st.session_state["fno_submitted_once"] = True
                    # proceed with saving trade logic below...

                required_fno_fields_filled = (
                        fno_symbol.strip() != ""
                        and margin_required > 0
                        and lot_size > 0
                        and num_lots > 0
                        and len(leg_data) > 0
                        and all(l["strike"] > 0 and l["premium"] > 0 for l in leg_data)
                )

                if not required_fno_fields_filled:
                    st.warning(
                        "⚠️ Please fill all required fields (symbol, capital, margin, strikes, premiums, and lot info).")
                    st.stop()

                image_path = None
                if uploaded_image is not None:
                    folder = Path("uploads") / str(trade_date)
                    folder.mkdir(parents=True, exist_ok=True)

                    filename = f"{fno_symbol.strip()}_{setup_name.strip()}_{wave_timeframe.strip()}_{strategy_type.strip()}.jpg".replace(
                        " ", "_")

                    image_path = folder / filename

                    try:
                        img = Image.open(uploaded_image)
                        img = img.convert("RGB")
                        img.save(image_path, format="JPEG", optimize=True, quality=65)
                    except Exception as e:
                        with open(image_path, "wb") as f:
                            f.write(uploaded_image.read())

                db = SessionLocal()
                try:
                    capital_used = capital_used if capital_used > 0 else margin_required + max(0, -net_credit)

                    header = TradeHeader(
                        trade_date=trade_date,
                        symbol=fno_symbol,
                        sentiment=sentiment,
                        setup=setup_name,
                        strategy=strategy_type,
                        capital_used=capital_used,
                        wave_timeframe=wave_timeframe,
                        margin_required=margin_required,
                        lot_size=lot_size,
                        num_lots=num_lots,
                        emotion=emotion,
                        confidence=confidence,
                        entry_price=spot_price,
                        stop_loss=spot_sl,
                        target=spot_target,
                        notes=notes,
                        image_path=str(image_path) if image_path else None,
                        is_closed=0,
                    )
                    db.add(header)
                    db.flush()

                    for leg in leg_data:
                        db.add(
                            TradeLeg(
                                trade_id=header.id,
                                label=leg["label"],
                                action=leg["action"],
                                option_type=leg["option_type"],
                                strike=leg["strike"],
                                premium=leg["premium"],
                                delta=leg["delta"],
                            )
                        )

                    db.commit()
                    st.success(f"✅ F&O trade saved successfully! Trade ID: {header.id}")
                    st.session_state["fno_submitted_once"] = False

                    if image_path:
                        st.image(str(image_path), caption="Uploaded Chart", use_container_width=True)
                except Exception as e:
                    db.rollback()
                    st.error(f"❌ Failed to save trade: {e}")
                finally:
                    db.close()

# ------------------------------------------------------------
# PAGE 2: VIEW TRADE LOGS
# ------------------------------------------------------------
elif page == "📋 View Trade Logs":
    st.title("📋 Trade Log Viewer")

    db = SessionLocal()
    trades = db.query(TradeHeader).order_by(TradeHeader.trade_date.desc()).all()
    db.close()

    if not trades:
        st.info("No trades logged yet.")
        st.stop()

    st.subheader("🔍 Filter Trades")

    df = pd.DataFrame(
        [
            {
                "Trade ID": str(t.id),
                "Date": t.trade_date,
                "Symbol": t.symbol,
                "Wave / Timeframe": getattr(t, "wave_timeframe", None),
                "Sentiment": t.sentiment,
                "Setup": t.setup,
                "Strategy": t.strategy,
                "Lot Size": t.lot_size,
                "Num Lots": t.num_lots,
                "Classification": t.classification,
                "Status": "Closed" if t.is_closed else "Open",
            }
            for t in trades
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input("Start Date", value=df["Date"].min().date())
    with col2:
        end_date = st.date_input("End Date", value=df["Date"].max().date())
    with col3:
        sentiment_filter = st.multiselect("Sentiment", sorted(df["Sentiment"].dropna().unique().tolist()))
    with col4:
        strategy_filter = st.multiselect("Strategy", sorted(df["Strategy"].dropna().unique().tolist()))

    search_text = st.text_input("Search Setup / Symbol / Notes")

    mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
    if sentiment_filter:
        mask &= df["Sentiment"].isin(sentiment_filter)
    if strategy_filter:
        mask &= df["Strategy"].isin(strategy_filter)
    if search_text:
        mask &= (
                df["Setup"].str.contains(search_text, case=False, na=False)
                | df["Symbol"].str.contains(search_text, case=False, na=False)
        )
    df_filtered = df[mask].copy()

    st.markdown("## 📈 Equity Trades Summary")
    df_equity = df_filtered[df_filtered["Strategy"].str.lower() == "equity"]

    if df_equity.empty:
        st.info("No Equity trades found for selected filters.")
    else:
        df_equity = df_equity.drop(columns=["Trade ID", "Num Lots"], errors="ignore")
        df_equity = df_equity.rename(columns={"Lot Size": "No. of Shares"})
        if "Wave / Timeframe" not in df_equity.columns:
            df_equity.insert(
                df_equity.columns.get_loc("Symbol") + 1,
                "Wave / Timeframe",
                df.loc[df_equity.index, "Wave / Timeframe"]
                if "Wave / Timeframe" in df.columns
                else None,
            )

        st.dataframe(df_equity, use_container_width=True)

    st.markdown("## 🔁 F&O Trades Summary")
    df_fno = df_filtered[df_filtered["Strategy"].str.lower() != "equity"]

    if df_fno.empty:
        st.info("No F&O trades found for selected filters.")
    else:
        df_fno = df_fno.drop(columns=["Trade ID"], errors="ignore")
        if "Wave / Timeframe" not in df_fno.columns:
            df_fno.insert(
                df_fno.columns.get_loc("Symbol") + 1,
                "Wave / Timeframe",
                df.loc[df_fno.index, "Wave / Timeframe"]
                if "Wave / Timeframe" in df.columns
                else None,
            )

        st.dataframe(df_fno, use_container_width=True)


    st.markdown(f"**Total Equity Trades:** {len(df_equity)}  |  **Total F&O Trades:** {len(df_fno)}")

    st.subheader("🔎 Trade Details (Expand for full view)")
    db = SessionLocal()
    for _, row in df_filtered.iterrows():
        trade = db.query(TradeHeader).filter_by(id=row["Trade ID"]).first()
        if not trade:
            continue

        with st.expander(f"{trade.trade_date} | {trade.symbol} | {trade.wave_timeframe or '—'} | {trade.strategy} | {trade.setup}"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sentiment", trade.sentiment)
            col2.metric("Capital Used (₹)", f"{trade.capital_used:,.2f}" if trade.capital_used else "—")
            col3.metric("Margin (₹)", f"{trade.margin_required:,.2f}" if trade.margin_required else "—")
            col4.metric("Confidence (%)", trade.confidence or 0)

            st.markdown(f"**Classification:** {trade.classification or '—'}")
            st.markdown(f"**Emotion:** {trade.emotion or '—'}")
            st.markdown(f"**Status:** {'Closed' if trade.is_closed else 'Open'}")

            if trade.image_path:
                st.image(trade.image_path, caption="📈 Entry Screenshot", use_container_width=True)
            if getattr(trade, "exit_image_path", None):
                st.image(trade.exit_image_path, caption="📉 Exit Screenshot", use_container_width=True)

            if trade.strategy.lower() == "equity":
                st.markdown("### 📈 Equity Trade Details")

                entry = getattr(trade, "entry_price", None)
                sl = getattr(trade, "stop_loss", None)
                exit_px = getattr(trade, "exit_price", None)

                if entry is None or exit_px is None:
                    legs = db.query(TradeLeg).filter_by(trade_id=trade.id).all()
                    if legs:
                        entry = entry if entry is not None else legs[0].premium
                        exit_px = exit_px if exit_px is not None else legs[0].exit_price

                if entry is None or sl is None or target is None or exit_px is None:
                    entry_match = re.search(r"Entry\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)", trade.notes or "", re.I)
                    sl_match = re.search(r"(?:SL|Stop\s*Loss)\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)", trade.notes or "", re.I)
                    target_match = re.search(r"Target\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)", trade.notes or "", re.I)
                    exit_match = re.search(r"Exit\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)", trade.notes or "", re.I)

                    if entry is None and entry_match:
                        entry = float(entry_match.group(1))
                    if sl is None and sl_match:
                        sl = float(sl_match.group(1))
                    if target is None and target_match:
                        target = float(target_match.group(1))
                    if exit_px is None and exit_match:
                        exit_px = float(exit_match.group(1))

                colE1, colE2, colE3, colE4 = st.columns(4)
                colE1.markdown(f"**Entry Price:** ₹{(entry or 0):.2f}")
                colE2.markdown(f"**Stop Loss:** ₹{(sl or 0):.2f}")
                colE3.markdown(f"**Target:** ₹{(target or 0):.2f}")
                colE4.markdown(f"**Exit Price:** ₹{(exit_px or 0):.2f}")

                if entry and sl and target:
                    risk = abs(entry - sl)
                    reward = abs(target - entry)
                    rr = round(reward / risk, 2) if risk else 0
                    st.info(f"📊 Risk–Reward: {rr}:1")

                if entry is not None and exit_px is not None:
                    pnl_per_unit = exit_px - entry
                    st.success(f"💵 P&L (per unit): ₹{pnl_per_unit:.2f}")
            else:
                st.markdown("### 🔁 F&O Legs Executed")
                legs = db.query(TradeLeg).filter_by(trade_id=trade.id).all()
                if legs:
                    leg_rows = []
                    total_credit = 0.0
                    total_debit = 0.0

                    for leg in legs:
                        pnl_leg = None
                        if leg.exit_price is not None:
                            lot_qty = (trade.lot_size or 1) * (trade.num_lots or 1)
                            if leg.action and leg.action.lower() == "buy":
                                pnl_leg = (leg.exit_price - leg.premium) * lot_qty
                            elif leg.action and leg.action.lower() == "sell":
                                pnl_leg = (leg.premium - leg.exit_price) * lot_qty

                        # track premiums
                        if leg.action.lower() == "sell":
                            total_credit += leg.premium
                        else:
                            total_debit += leg.premium

                        leg_rows.append({
                            "Leg Label": leg.label,
                            "Action": leg.action,
                            "Option Type": leg.option_type,
                            "Strike": leg.strike,
                            "Entry Premium": leg.premium,
                            "Exit Price": leg.exit_price,
                            "Δ (Delta)": leg.delta,
                            "Leg PnL": pnl_leg
                        })

                    df_legs = pd.DataFrame(leg_rows)
                    st.dataframe(df_legs, use_container_width=True)
                    # --- Calculate spread metrics (BEP, Max Profit, Max Loss, Risk @ SL) ---
                    metrics = calculate_spread_metrics(
                        trade.strategy,
                        legs,
                        trade.lot_size or 1,
                        trade.num_lots or 1,
                        stop_loss_spot=trade.stop_loss,
                        entry_spot=trade.entry_price,
                    )

                    if metrics:
                        colA, colB, colC, colD = st.columns(4)
                        colA.metric("BEP (Approx)", metrics["bep"])
                        colB.metric("Max Profit (₹)", metrics["max_profit"])
                        colC.metric("Max Risk (₹)", metrics["max_loss"])
                        if metrics["risk_at_sl"]:
                            colD.metric("Risk @ SL Hit (₹)", metrics["risk_at_sl"])

                    # Compute average absolute delta across legs
                    leg_deltas = [abs(l.delta) for l in legs if l.delta is not None]
                    avg_delta = sum(leg_deltas) / len(leg_deltas) if leg_deltas else 0.3


                    # ---- Calculate derived metrics ----
                    lot_qty = (trade.lot_size or 1) * (trade.num_lots or 1)
                    net_credit = (total_credit - total_debit) * lot_qty

                    # --- BEP Calculation ---
                    strikes = [l.strike for l in legs if l.strike > 0]
                    breakeven = None
                    if len(strikes) == 2:
                        if "Put" in trade.strategy:
                            breakeven = strikes[0] - abs(net_credit / lot_qty)
                        elif "Call" in trade.strategy:
                            breakeven = strikes[0] + abs(net_credit / lot_qty)

                    # --- Max Risk Calculation ---
                    if net_credit > 0:
                        max_risk = abs((trade.margin_required or 0) - abs(net_credit))
                    else:
                        max_risk = abs(net_credit)

                    # --- Compute from legs instead of header ---
                    leg_deltas = [abs(l.delta) for l in legs if l.delta is not None]
                    avg_delta = sum(leg_deltas) / len(leg_deltas) if leg_deltas else 0.3

                    # --- Risk When SL Hits (Spot-based estimate) ---
                    max_risk_at_sl = None
                    max_profit = None
                    if trade.stop_loss and trade.entry_price:
                        spot_move = abs(trade.entry_price - trade.stop_loss)
                        lot_qty = (trade.lot_size or 1) * (trade.num_lots or 1)
                        approx_loss_per_point = lot_qty * avg_delta
                        est_loss = spot_move * approx_loss_per_point
                        max_risk_at_sl = min(est_loss, max_risk)

                    # --- Maximum Profit (approx) ---
                    if net_credit > 0:
                        max_profit = abs(net_credit)  # for credit spreads
                    else:
                        max_profit = abs(net_credit)  # for debit spreads (profit capped by net debit)



                    # ---- PnL + ROI ----
                    total_pnl = sum([x["Leg PnL"] for x in leg_rows if x["Leg PnL"] is not None])
                    if total_pnl is not None:
                        st.success(f"💰 Total Strategy P&L: ₹{total_pnl:,.2f}")
                        if trade.capital_used:
                            roi = (total_pnl / trade.capital_used) * 100
                            st.info(f"📈 ROI: {roi:.2f}%")

            if trade.notes:
                st.markdown("### 📝 Notes")
                st.write(trade.notes)
    db.close()

    st.markdown("## 📘 Download Detailed Trades Report (Filtered)")

    if st.button("📥 Generate Detailed Report (Excel)"):
        db = SessionLocal()
        trade_ids = df_filtered["Trade ID"].tolist()
        trades_filtered = db.query(TradeHeader).filter(TradeHeader.id.in_(trade_ids)).all()

        equity_rows, fno_rows, fno_leg_rows = [], [], []

        for t in trades_filtered:
            base_info = {
                "Trade ID": str(t.id),
                "Date": t.trade_date,
                "Symbol": t.symbol,
                "Wave / Timeframe": getattr(t, "wave_timeframe", None),  # ✅ Added here
                "Sentiment": t.sentiment,
                "Setup": t.setup,
                "Strategy": t.strategy,
                "Capital Used (₹)": t.capital_used,
                "Margin Required (₹)": t.margin_required,
                "Lot Size": t.lot_size,
                "Num Lots": t.num_lots,
                "Emotion": t.emotion,
                "Confidence (%)": t.confidence,
                "Classification": t.classification,
                "Status": "Closed" if t.is_closed else "Open",
                "Notes": t.notes,
            }

            if t.strategy.lower() == "equity":
                # 1) Read from header (primary)
                entry_val = getattr(t, "entry_price", None)
                sl_val = getattr(t, "stop_loss", None)
                target_val = getattr(t, "target", None)
                exit_val = getattr(t, "exit_price", None)

                if not entry_val or not exit_val:
                    legs = db.query(TradeLeg).filter_by(trade_id=t.id).all()
                    if legs:
                        leg = legs[0]
                        entry_val = entry_val or leg.premium
                        exit_val = exit_val or leg.exit_price

                if not entry_val or not sl_val or not target_val or not exit_val:
                    entry = re.search(r"Entry[:=-]\s*([0-9.]+)", t.notes or "", re.I)
                    sl = re.search(r"(?:SL|Stop\s*Loss)[:=-]\s*([0-9.]+)", t.notes or "", re.I)
                    target = re.search(r"Target[:=-]\s*([0-9.]+)", t.notes or "", re.I)
                    exit_ = re.search(r"Exit[:=-]\s*([0-9.]+)", t.notes or "", re.I)
                    if entry and not entry_val:
                        entry_val = float(entry.group(1))
                    if sl and not sl_val:
                        sl_val = float(sl.group(1))
                    if target and not target_val:
                        target_val = float(target.group(1))
                    if exit_ and not exit_val:
                        exit_val = float(exit_.group(1))

                rr_ratio = None
                if entry_val and sl_val and target_val:
                    risk = abs(entry_val - sl_val)
                    reward = abs(target_val - entry_val)
                    rr_ratio = round(reward / risk, 2) if risk else None

                pnl = None
                if entry_val is not None and exit_val is not None:
                    pnl = exit_val - entry_val

                equity_rows.append({
                    **base_info,
                    "Entry Price (₹)": entry_val,
                    "Stop Loss (₹)": sl_val,
                    "Target (₹)": target_val,
                    "Exit Price (₹)": exit_val,
                    "Risk–Reward": rr_ratio,
                    "PnL (per unit ₹)": pnl
                })

            else:
                fno_rows.append(base_info)

                legs = db.query(TradeLeg).filter_by(trade_id=t.id).all()
                for leg in legs:
                    pnl_leg = None
                    if leg.exit_price is not None:
                        lot_qty = (t.lot_size or 1) * (t.num_lots or 1)
                        if leg.action and leg.action.lower() == "buy":
                            pnl_leg = (leg.exit_price - leg.premium) * lot_qty
                        elif leg.action and leg.action.lower() == "sell":
                            pnl_leg = (leg.premium - leg.exit_price) * lot_qty

                    fno_leg_rows.append({
                        "Trade ID": str(t.id),
                        "Date": t.trade_date,
                        "Strategy": t.strategy,
                        "Leg Label": leg.label,
                        "Action": leg.action,
                        "Option Type": leg.option_type,
                        "Strike (₹)": leg.strike,
                        "Entry Premium (₹)": leg.premium,
                        "Exit Premium (₹)": leg.exit_price,
                        "Leg PnL (₹)": pnl_leg
                    })

        db.close()

        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            if equity_rows:
                pd.DataFrame(equity_rows).to_excel(writer, index=False, sheet_name="Equity_Trades")
            if fno_rows:
                pd.DataFrame(fno_rows).to_excel(writer, index=False, sheet_name="FNO_Trades")
            if fno_leg_rows:
                pd.DataFrame(fno_leg_rows).to_excel(writer, index=False, sheet_name="FNO_Legs")

        excel_buffer.seek(0)
        st.download_button(
            label="💾 Download Detailed Report (Excel)",
            data=excel_buffer,
            file_name=f"Trade_Report_{start_date}_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with st.expander("📊 Strategy Distribution"):
        strat_counts = df["Strategy"].value_counts().reset_index()
        strat_counts.columns = ["Strategy", "Trades"]
        fig = px.pie(strat_counts, names="Strategy", values="Trades", title="Strategy Usage Share", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# PAGE 3: EXIT / UPDATE TRADES
# ------------------------------------------------------------
elif page == "📈 Exit / Update Trades":
    st.title("📈 Exit or Update Open Trades")

    db = SessionLocal()
    open_trades = db.query(TradeHeader).filter_by(is_closed=0).all()
    if not open_trades:
        st.info("No open trades found.")
        db.close()
        st.stop()

    trade_map = {
        f"{t.trade_date} | {t.symbol} | {t.wave_timeframe or '—'} | {t.strategy} | {t.setup}": t
        for t in open_trades
    }

    selected_key = st.selectbox("Select Trade", list(trade_map.keys()))
    selected_trade = trade_map[selected_key]

    st.markdown(f"### {selected_trade.symbol} ({selected_trade.strategy})")
    st.write(f"**Setup:** {selected_trade.setup}")
    st.write(f"**Sentiment:** {selected_trade.sentiment}")
    st.write(f"**Status:** {'Closed' if selected_trade.is_closed else 'Open'}")

    if selected_trade.strategy.lower() == "equity":
        exit_price = st.number_input(
            "Exit Price (Spot / CMP)",
            min_value=0.0,
            step=0.05,
            format="%.2f",
            key=f"equity_exit_{selected_trade.id}",
        )
        uploaded_exit_image = st.file_uploader(
            "📷 Upload Exit Screenshot (optional)",
            type=["png", "jpg", "jpeg"],
            key=f"exit_img_equity_{selected_trade.id}",
        )

        classification = st.selectbox(
            "Trade Classification", ["Good Gain", "Bad Gain", "Good Loss", "Bad Loss"]
        )
        remarks = st.text_area("Final Remarks")

        if st.button("💾 Update Equity Trade"):
            try:
                leg = db.query(TradeLeg).filter_by(trade_id=selected_trade.id).first()
                if leg:
                    leg.exit_price = exit_price
                else:
                    leg = TradeLeg(trade_id=selected_trade.id, label="Equity Position", exit_price=exit_price)
                    db.add(leg)

                selected_trade.exit_price = exit_price
                selected_trade.classification = classification
                selected_trade.is_closed = 1
                if remarks:
                    selected_trade.notes = (selected_trade.notes or "") + "\n" + remarks

                # ✅ Save exit screenshot (compressed)
                exit_image_path = None
                if uploaded_exit_image is not None:
                    try:
                        entry_folder = Path(selected_trade.image_path).parent if getattr(selected_trade, "image_path", None) else Path("uploads")

                        entry_folder.mkdir(parents=True, exist_ok=True)

                        exit_filename = Path(
                            selected_trade.image_path).stem + "_exit.jpg" if selected_trade.image_path else f"{selected_trade.symbol}_exit.jpg"
                        exit_image_path = entry_folder / exit_filename

                        img = Image.open(uploaded_exit_image).convert("RGB")
                        img.save(exit_image_path, format="JPEG", optimize=True, quality=65)
                        selected_trade.exit_image_path = str(exit_image_path)
                    except Exception as e:
                        st.warning(f"⚠️ Could not save exit screenshot: {e}")

                db.flush()   # ensure all updates are written
                db.commit()
                st.success("✅ Equity trade updated and closed successfully (exit price saved).")
            except Exception as e:
                db.rollback()
                st.error(f"❌ Error: {str(e)}")
            finally:
                db.close()

    # ========================================================
    # F&O TRADES
    # ========================================================
    else:
        legs = db.query(TradeLeg).filter_by(trade_id=selected_trade.id).all()

        if not legs:
            st.warning("No legs found for this trade.")
            db.close()
            st.stop()

        # 🟩 Exit Spot Section
        exit_spot_price = st.number_input(
            "Exit Spot Price (₹)",
            min_value=0.0,
            step=0.05,
            format="%.2f",
            key=f"exit_spot_{selected_trade.id}",
        )
        uploaded_exit_image = st.file_uploader(
            "📷 Upload Exit Screenshot (optional)",
            type=["png", "jpg", "jpeg"],
            key=f"exit_img_fno_{selected_trade.id}",
        )


        st.markdown("### Leg-wise Exit Prices")
        per_leg_exits = []

        for leg in legs:
            exit_val = st.number_input(
                f"Exit Premium – {leg.label} ({leg.action} {leg.option_type} @ {leg.strike})",
                min_value=0.0,
                step=0.05,
                format="%.2f",
                key=f"exit_{leg.id}",
            )
            per_leg_exits.append({"leg_id": leg.id, "exit_price": exit_val})

        classification = st.selectbox(
            "Trade Classification", ["Good Gain", "Bad Gain", "Good Loss", "Bad Loss"]
        )
        remarks = st.text_area("Final Remarks / Observations")

        if st.button("💾 Update F&O Trade"):
            try:
                for leg_exit in per_leg_exits:
                    db.query(TradeLeg).filter_by(id=leg_exit["leg_id"]).update(
                        {"exit_price": leg_exit["exit_price"]}
                    )

                selected_trade.classification = classification
                selected_trade.exit_price = exit_spot_price

                selected_trade.is_closed = 1
                if remarks:
                    selected_trade.notes = (selected_trade.notes or "") + "\n" + remarks
                # ✅ Save exit screenshot (compressed)
                exit_image_path = None
                if uploaded_exit_image is not None:
                    try:
                        entry_folder = Path(selected_trade.image_path).parent if getattr(selected_trade, "image_path", None) else Path("uploads")

                        entry_folder.mkdir(parents=True, exist_ok=True)

                        exit_filename = Path(
                            selected_trade.image_path).stem + "_exit.jpg" if selected_trade.image_path else f"{selected_trade.symbol}_exit.jpg"
                        exit_image_path = entry_folder / exit_filename

                        img = Image.open(uploaded_exit_image).convert("RGB")
                        img.save(exit_image_path, format="JPEG", optimize=True, quality=65)
                        selected_trade.exit_image_path = str(exit_image_path)
                    except Exception as e:
                        st.warning(f"⚠️ Could not save exit screenshot: {e}")

                db.flush()   # forces ORM to push updates
                db.commit()
                st.success("✅ F&O trade updated successfully — all legs closed and synced.")
            except Exception as e:
                db.rollback()
                st.error(f"❌ Error: {str(e)}")
            finally:
                db.close()


# ------------------------------------------------------------
# PAGE 4: CODE OF CONDUCT (Stylized + Compatible with all Streamlit versions)
# ------------------------------------------------------------
elif page == "📜 Code of Conduct":
    html_content = """
    <style>
    body {
        background-color: #f9f9fb;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    .conduct-container {
        background-color: #ffffff;
        padding: 2rem 3rem;
        border-radius: 18px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.08);
        margin-top: 1rem;
        color: #222;
        line-height: 1.7;
    }
    .conduct-title {
        text-align: center;
        color: #a80000;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .conduct-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .conduct-point {
        padding: 0.4rem 0;
        font-size: 1.05rem;
        border-left: 3px solid #d32f2f;
        padding-left: 12px;
        margin-bottom: 0.4rem;
    }
    .conduct-point::before {
        content: "💎 ";
        color: #b71c1c;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-style: italic;
        color: #444;
        font-size: 1.05rem;
    }
    </style>

    <div class="conduct-container">
        <div class="conduct-title">Code of Conduct</div>
        <div class="conduct-subtitle">A daily affirmation for discipline, gratitude, and trading mastery</div>

        <div class="conduct-point">Market is God and am very Grateful to Market for giving opportunities to succeed.</div>
        <div class="conduct-point">I follow the market price action and not my Mind.</div>
        <div class="conduct-point">I trade only high probability setups.</div>
        <div class="conduct-point">I always mind my entry.</div>
        <div class="conduct-point">I analyse charts as if I have no position in them.</div>
        <div class="conduct-point">I consistently follow my trading plans.</div>
        <div class="conduct-point">I am accountable for my trading.</div>
        <div class="conduct-point">I keep my trading diary with performance matrix.</div>
        <div class="conduct-point">I am always with the dominant force in the market.</div>
        <div class="conduct-point">I believe in my trading strategies completely and whole heartedly.</div>
        <div class="conduct-point">I always follow what the market is doing and not what I want market to do.</div>
        <div class="conduct-point">I practice proper risk management, I let the profit run and take only small losses.</div>
        <div class="conduct-point">I take only those trades that gives me reward which clearly outweighs the risk.</div>
        <div class="conduct-point">I find other things to do once my trades go live.</div>
        <div class="conduct-point">I am not emotionally affected by profits/losses.</div>
        <div class="conduct-point">I count my profit only after exiting my trades.</div>
        <div class="conduct-point">I continue to stay in my trades till I find a good cause to exit.</div>
        <div class="conduct-point">I am happy and satisfied with every profit.</div>
        <div class="conduct-point">I respect money and I always protect my capital.</div>
        <div class="conduct-point">I am detached from news and fundamentals while trading.</div>
        <div class="conduct-point">I am the most disciplined trader ever.</div>
        <div class="conduct-point">Market is God and am very Grateful to Market for giving opportunities to succeed.</div>
        <div class="conduct-point">I am happy and a successful trader!!!</div>

        <div class="footer">💫 Stay Humble · Stay Patient · Stay Consistent · Be Grateful 💫</div>
    </div>
    """

    # ✅ Proper HTML rendering
    components.html(html_content, height=1100, scrolling=True)
