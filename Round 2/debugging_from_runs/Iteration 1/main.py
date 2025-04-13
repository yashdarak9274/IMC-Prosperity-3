from typing import Dict, List
from datamodel import Order, TradingState 
import numpy as np
import json

POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
}

class Trader:
    def __init__(self):
        self.history = {}
        self.risk_adjustment = 0.8
        self.max_spread = 5

    def run(self, state: TradingState):
        result = {}
        self._load_history(state.traderData)

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []
            
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
            else:
                mid_price = self._get_last_price(product)
                spread = self.max_spread

            position = state.position.get(product, 0)
            self._update_history(product, mid_price)

            if product == "RAINFOREST_RESIN":
                orders = self._market_make(product, mid_price, position, spread)
            elif product == "KELP":
                orders = self._mean_reversion(product, position, window=10)
            elif product == "SQUID_INK":
                orders = self._dynamic_zscore_arbitrage(product, position, window=15)

            result[product] = orders

        return result, 0, json.dumps(self.history)

    def _load_history(self, trader_data):
        try:
            self.history = json.loads(trader_data) if trader_data else {}
        except:
            self.history = {}

    def _update_history(self, product, price):
        if product not in self.history:
            self.history[product] = []
        self.history[product].append(price)
        if len(self.history[product]) > 100:
            self.history[product].pop(0)

    def _get_last_price(self, product):
        return self.history[product][-1] if self.history.get(product) else 10000

    # Adaptive Market Making Strategy for RAINFOREST_RESIN
    def _market_make(self, product, mid_price, position, spread):
        recent_prices = self.history[product][-10:]
        volatility = np.std(recent_prices) if recent_prices else 1
        dynamic_spread = min(volatility * 1.5, self.max_spread)
        bid_price = int(mid_price - dynamic_spread / 2)
        ask_price = int(mid_price + dynamic_spread / 2)

        max_buy = POSITION_LIMITS[product] - position
        max_sell = POSITION_LIMITS[product] + position

        orders = []
        if max_buy > 0:
            orders.append(Order(product, bid_price, max_buy))
        if max_sell > 0:
            orders.append(Order(product, ask_price, -max_sell))

        return orders

    # Improved Mean Reversion for KELP
    def _mean_reversion(self, product, position, window=10):
        if len(self.history.get(product, [])) < window:
            return []
            
        hist_prices = self.history[product][-window:]
        ma = np.mean(hist_prices)
        volatility = np.std(hist_prices)
        current_price = hist_prices[-1]

        orders = []
        threshold = volatility * 0.5

        if current_price < ma - threshold:
            volume = POSITION_LIMITS[product] - position
            orders.append(Order(product, int(current_price), volume))
        elif current_price > ma + threshold:
            volume = POSITION_LIMITS[product] + position
            orders.append(Order(product, int(current_price), -volume))

        return orders

    # Dynamic Z-Score Arbitrage for SQUID_INK
    def _dynamic_zscore_arbitrage(self, product, position, window=15):
        if len(self.history.get(product, [])) < window:
            return []
            
        prices = np.array(self.history[product][-window:])
        mean_price = np.mean(prices)
        std_price = np.std(prices) + 1e-5
        z_score = (prices[-1] - mean_price) / std_price

        recent_volatility = np.std(prices[-5:])  # Recent volatility measure
        dynamic_threshold = max(0.8, min(1.5, 1.0 * (recent_volatility / std_price)))

        orders = []
        if z_score < -dynamic_threshold:
            volume = POSITION_LIMITS[product] - position
            orders.append(Order(product, int(prices[-1]), volume))
        elif z_score > dynamic_threshold:
            volume = POSITION_LIMITS[product] + position
            orders.append(Order(product, int(prices[-1]), -volume))

        return orders
