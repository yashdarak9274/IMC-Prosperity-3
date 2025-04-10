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
                orders = self._mean_reversion(product, position, window=15)
            elif product == "SQUID_INK":
                orders = self._statistical_arbitrage(product, position, window=25)

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

    def _market_make(self, product, mid_price, position, spread):
        orders = []
        dynamic_spread = min(spread * self.risk_adjustment, self.max_spread)
        bid_price = int(mid_price - dynamic_spread/2)
        ask_price = int(mid_price + dynamic_spread/2)
        
        max_buy = POSITION_LIMITS[product] - position
        max_sell = POSITION_LIMITS[product] + position
        
        if max_buy > 0:
            orders.append(Order(product, bid_price, max_buy))
        if max_sell > 0:
            orders.append(Order(product, ask_price, -max_sell))
            
        return orders

    def _mean_reversion(self, product, position, window=15):
        if len(self.history.get(product, [])) < window:
            return []
            
        hist_prices = self.history[product][-window:]
        ma = np.mean(hist_prices)
        current_price = hist_prices[-1]
        spread = np.std(hist_prices) * 1.5
        
        if current_price < ma - spread/2:
            return [Order(product, int(current_price), POSITION_LIMITS[product] - position)]
        elif current_price > ma + spread/2:
            return [Order(product, int(current_price), -POSITION_LIMITS[product] - position)]
        return []

    def _statistical_arbitrage(self, product, position, window=25):
        if len(self.history.get(product, [])) < window:
            return []
            
        prices = np.array(self.history[product][-window:])
        z_score = (prices[-1] - np.mean(prices)) / np.std(prices)
        
        if z_score < -1.5:
            return [Order(product, int(prices[-1]), POSITION_LIMITS[product] - position)]
        elif z_score > 1.5:
            return [Order(product, int(prices[-1]), -POSITION_LIMITS[product] - position)]
        return []
