# app/utils/strategy_mapping.py

STRATEGY_MAPPING = {
    "Bullish Swing": {
        "setups": [
            "Fake Breakdown",
            "Double Bottom",
            "Bull Counter Attack",
            "Mother Candle Bullish Reversal",
            "Three White Soldiers",
            "Sandwitch Bullish Reversal",
            "Morning Star",
            "Bullish Engulf",
            "Last Bearish Engulf"
        ],
        "strategies": [
            "Covered Call",
            "Bull Put Spread",
            "Bull Put Ladder",
            "Bull Collar Long"
        ]
    },

    "Bullish Momentum": {
        "setups": [
            "Genuine Breakout",
            "Flag N Pole",
            "Cup and Handle",
            "Mother Candle Bullish Continuation",
            "Sandwitch Bullish Continuation",
            "Gap Up Above Resistance"
        ],
        "strategies": [
            "Bull Call Spread",
            "Bull Call Ratio Spread",
            "Bull Calendar Spread",
            "Covered Call",
            "Bull Put Spread"
        ]
    },

    "Bearish Swing": {
        "setups": [
            "Fake Breakout",
            "Double Top",
            "Bear Counter Attack",
            "Mother Candle Bearish Reversal",
            "Three Black Crows",
            "Sandwitch Bearish Reversal",
            "Evening Star",
            "Bearish Engulf",
            "Last Bullish Engulf",
            "Dark Cloud Cover"
        ],
        "strategies": [
            "Covered Put",
            "Bear Call Spread",
            "Bear Call Ladder",
            "Bear Collar Short"
        ]
    },

    "Bearish Momentum": {
        "setups": [
            "Genuine Breakdown",
            "Flag N Pole",
            "Rounding Top",
            "Mother Candle Bearish Continuation",
            "Sandwitch Bearish Continuation",
            "Gap Down Below Support"
        ],
        "strategies": [
            "Bear Put Spread",
            "Bear Put Ratio Spread",
            "Bear Calendar Spread",
            "Covered Put",
            "Bear Call Spread"
        ]
    },

    "Sideways": {
        "setups": ["Range Bound"],
        "strategies": [
            "Short Iron Condor",
            "Short Iron Butterfly",
            "Short Strangle",
            "Short Straddle"
        ]
    },

    "Event Days": {
        "setups": ["Event Day"],
        "strategies": [
            "Long Iron Condor",
            "Long Iron Butterfly",
            "Long Strangle",
            "Long Straddle"
        ]
    }
}


def get_sentiment_types():
    return list(STRATEGY_MAPPING.keys())


def get_setups_by_sentiment(sentiment: str):
    return STRATEGY_MAPPING.get(sentiment, {}).get("setups", [])


def get_strategies_by_sentiment(sentiment: str):
    return STRATEGY_MAPPING.get(sentiment, {}).get("strategies", [])
