# indicator_config.py

INDICATOR_SETUP_MAP = {
    "Bullish Swing": {
        "TIDE": {
            "MACD": ["uptick", "pco", "flat after down"]
        },
        "WAVE": {
            "MACD": ["uptick", "pco"],
            "RSI": { "condition": ">", "value": 40 },
            "STOCHASTICS": ["pco"],
            "BOLLINGER": ["bkp"],
            "VOLUME": { "condition": ">", "value": "average" },
            "DIVERGENCE": ["bullish"]
        }
    },

    "Bullish Momentum": {
        "TIDE": {
            "MACD": ["uptick", "pco", "flat after down"]
        },
        "WAVE": {
            "MACD": ["uptick", "pco", "flat after down"],
            "RSI": { "condition": ">", "value": 40 },
            "STOCHASTICS": ["pco in overbought zone"],
            "BOLLINGER": ["upper bb challenged"],
            "VOLUME": { "condition": ">", "value": "average" },
            "FIB": [23, 38, 50, 62, 78],
            "EMA": ["5>13>26"]
        }
    },

    "Bearish Swing": {
        "TIDE": { "MACD": ["flat after up", "downtick", "nco"] },
        "WAVE": { "MACD": ["downtick", "nco"], "RSI": { "condition": "<", "value": 60 },
                  "STOCHASTICS": ["nco"], "BOLLINGER": ["upper bb failed (bkt)"],
                  "VOLUME": { "condition": ">", "value": "average" },
                  "DIVERGENCE": ["evening star", "double top"] }
    },

    "Bearish Momentum": {
        "TIDE": { "MACD": ["downtick", "nco"] },
        "WAVE": { "MACD": ["downtick", "nco"], "RSI": { "condition": "<", "value": 40 },
                  "STOCHASTICS": ["nco in oversold"], "BOLLINGER": ["bbc down"],
                  "VOLUME": { "condition": ">", "value": "average" },
                  "FIB": [23, 38, 50, 62, 78], "EMA": ["26>13>5"] }
    },

    "Sideways": {
        "TIDE": { "MACD": ["flat", "neutral"] },
        "WAVE": { "MACD": ["toggling near 0"], "RSI": { "condition": "between", "value": [40, 60] },
                  "STOCHASTICS": ["bw overbought and oversold"],
                  "BOLLINGER": ["flat bb"], "VOLUME": ["average"] }
    }
}
