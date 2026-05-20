import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_price_data(
    n_days=365,
    start_date="2024-01-01"
):
    np.random.seed(42)
    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    products = {
        "fish_sauce":     {"base_price": 3.20, "volatility": 0.05},
        "miso_paste":     {"base_price": 4.50, "volatility": 0.03},
        "soy_sauce":      {"base_price": 2.80, "volatility": 0.04},
        "coconut_milk":   {"base_price": 1.90, "volatility": 0.06},
        "olive_oil":      {"base_price": 12.00, "volatility": 0.08},
        "eggs_10":        {"base_price": 3.20, "volatility": 0.07},
        "rice_5kg":       {"base_price": 8.50, "volatility": 0.03},
        "chicken_breast": {"base_price": 6.80, "volatility": 0.09},
        "greek_yogurt":   {"base_price": 4.20, "volatility": 0.05},
        "canned_tomato":  {"base_price": 1.50, "volatility": 0.04},
    }

    for product, config in products.items():
        base = config["base_price"]
        vol = config["volatility"]

        for day in range(n_days):
            date = start + timedelta(days=day)
            trend = base * (1 + 0.003 * day / 30)
            seasonal = base * 0.05 * np.sin(2 * np.pi * day / 365)
            noise = np.random.normal(0, base * vol)

            spike = 0
            is_anomaly = 0
            if np.random.random() < 0.03:
                spike = base * np.random.uniform(0.15, 0.40)
                is_anomaly = 1

            price = round(max(0.1, trend + seasonal + noise + spike), 2)

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "product": product,
                "price": price,
                "is_anomaly": is_anomaly
            })

    df = pd.DataFrame(records)
    df.to_csv("data/price_history.csv", index=False)
    print(f"Generated {len(df)} price records for {len(products)} products")
    return df

if __name__ == "__main__":
    generate_price_data()