# app/utils/option_calculator.py

def calculate_strategy_metrics(strategy, legs):
    try:
        if strategy == "Bull Put Spread":
            short_put = next(l for l in legs if l["action"] == "Sell" and l["option_type"] == "PUT")
            long_put = next(l for l in legs if l["action"] == "Buy" and l["option_type"] == "PUT")
            net_premium = short_put["premium"] - long_put["premium"]
            spread = short_put["strike"] - long_put["strike"]
            max_profit = net_premium * 100
            max_loss = (spread - net_premium) * 100
            bep = short_put["strike"] - net_premium
            return {"net_premium": net_premium, "max_profit": max_profit, "max_loss": max_loss, "bep": bep}

        elif strategy == "Bull Call Ratio Spread":
            atm_calls = [l for l in legs if l["action"] == "Buy" and l["option_type"] == "CALL"]
            otm_call = next(l for l in legs if l["action"] == "Sell" and l["option_type"] == "CALL")
            net_premium = sum(l["premium"] for l in atm_calls) - otm_call["premium"]
            return {"net_premium": net_premium, "max_profit": "Variable", "max_loss": "Limited", "bep": "Depends on strikes"}

        # Add more strategies similarly...

        return {"net_premium": 0.0}
    except Exception as e:
        print(f"Error in calculation: {e}")
        return {"net_premium": 0.0}
