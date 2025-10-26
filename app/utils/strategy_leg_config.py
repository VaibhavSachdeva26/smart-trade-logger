# app/utils/strategy_leg_config.py

STRATEGY_LEG_CONFIG = {
    # 🔵 Bullish Strategies
    "Bull Put Spread": [
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (ATM)"},
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT (OTM)"}
    ],
    "Bull Call Spread": [
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL (ATM)"},
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (OTM)"}
    ],
    "Bull Put Ladder": [
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (ATM)"},
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT (Lower OTM)"},
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT (Further OTM)"}
    ],
    "Bull Call Ladder": [
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL (ATM)"},
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (Higher)"},
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (Further Higher)"}
    ],
    "Bull Collar Long": [
        {"action": "Buy", "option_type": "STOCK", "label": "Buy Underlying"},
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT (Protection)"},
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (Cap)"}
    ],
    "Bull Call Ratio Spread": [
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL 1 (ATM)"},
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL 2 (ATM)"},
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (OTM)"}
    ],

    # 🔴 Bearish Strategies
    "Bear Call Spread": [
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (ATM)"},
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL (OTM)"}
    ],
    "Bear Put Spread": [
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT (ATM)"},
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (OTM)"}
    ],
    "Bear Call Ladder": [
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (ATM)"},
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL (Higher)"},
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL (Further Higher)"}
    ],
    "Bear Put Ladder": [
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT (ATM)"},
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (Lower)"},
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (Further Lower)"}
    ],
    "Bear Collar Short": [
        {"action": "Sell", "option_type": "STOCK", "label": "Short Underlying"},
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL (Protection)"},
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (Cap)"}
    ],
    "Bear Put Ratio Spread": [
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT 1 (ATM)"},
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT 2 (ATM)"},
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (OTM)"}
    ],

    # ⚖️ Neutral / Range-Bound
    "Short Straddle": [
        {"action": "Sell", "option_type": "CALL", "label": "Short ATM CALL"},
        {"action": "Sell", "option_type": "PUT", "label": "Short ATM PUT"}
    ],
    "Short Strangle": [
        {"action": "Sell", "option_type": "CALL", "label": "Short OTM CALL"},
        {"action": "Sell", "option_type": "PUT", "label": "Short OTM PUT"}
    ],
    "Short Iron Condor": [
        {"action": "Buy", "option_type": "PUT", "label": "Long lower PUT"},
        {"action": "Sell", "option_type": "PUT", "label": "Short middle PUT"},
        {"action": "Sell", "option_type": "CALL", "label": "Short middle CALL"},
        {"action": "Buy", "option_type": "CALL", "label": "Long higher CALL"}
    ],
    "Short Iron Butterfly": [
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT (lower strike)"},
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (ATM)"},
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (ATM)"},
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL (higher strike)"}
    ],

    # 🔁 Long Volatility (Event Days)
    "Long Straddle": [
        {"action": "Buy", "option_type": "CALL", "label": "Long ATM CALL"},
        {"action": "Buy", "option_type": "PUT", "label": "Long ATM PUT"}
    ],
    "Long Strangle": [
        {"action": "Buy", "option_type": "CALL", "label": "Long OTM CALL"},
        {"action": "Buy", "option_type": "PUT", "label": "Long OTM PUT"}
    ],
    "Long Iron Condor": [
        {"action": "Sell", "option_type": "PUT", "label": "Short lower PUT"},
        {"action": "Buy", "option_type": "PUT", "label": "Long middle PUT"},
        {"action": "Buy", "option_type": "CALL", "label": "Long middle CALL"},
        {"action": "Sell", "option_type": "CALL", "label": "Short higher CALL"}
    ],
    "Long Iron Butterfly": [
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (lower strike)"},
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT (ATM)"},
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL (ATM)"},
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (higher strike)"}
    ],

    # 📅 Calendar / Time Spread
    "Calendar Spread (CALL)": [
        {"action": "Sell", "option_type": "CALL", "label": "Short CALL (near expiry)"},
        {"action": "Buy", "option_type": "CALL", "label": "Long CALL (far expiry)"}
    ],
    "Calendar Spread (PUT)": [
        {"action": "Sell", "option_type": "PUT", "label": "Short PUT (near expiry)"},
        {"action": "Buy", "option_type": "PUT", "label": "Long PUT (far expiry)"}
    ],

    # 📈 Covered Strategies
    "Covered Call": [
        {"action": "Buy", "option_type": "STOCK", "label": "Buy Underlying"},
        {"action": "Sell", "option_type": "CALL", "label": "Sell OTM CALL"}
    ],
    "Covered Put": [
        {"action": "Sell", "option_type": "STOCK", "label": "Sell Underlying"},
        {"action": "Sell", "option_type": "PUT", "label": "Sell OTM PUT"}
    ]
}
