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

    def run(self, state: TradingState):
        result = {}

        # Load historical data from traderData string
        if state.traderData and state.traderData != "Yash":
            try:
                self.history = json.loads(state.traderData)
            except:
                self.history = {}
        else:
            self.history = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []

            # Skip products without both buy and sell orders
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2

            position = state.position.get(product, 0)

            # Initialize price history list if not present
            if product not in self.history:
                self.history[product] = []
            self.history[product].append(mid_price)

            # Your custom trading logic
            if product == "RAINFOREST_RESIN":
                orders += self.trade_resin(mid_price, position)
            elif product == "KELP":
                orders += self.trade_kelp(mid_price, position, product)
            elif product == "SQUID_INK":
                orders += self.trade_squid_ink(mid_price, position, product)

            result[product] = orders

        conversions = 0 
        # Properly persist the history as JSON
        traderData = json.dumps(self.history)

        return result, conversions, traderData

    def trade_resin(self, mid_price, position):
        orders = []
        size = 5
        # Round prices to integers
        bid = int(mid_price - 1)
        ask = int(mid_price + 1)

        max_buy = POSITION_LIMITS["RAINFOREST_RESIN"] - position
        max_sell = POSITION_LIMITS["RAINFOREST_RESIN"] + position

        if max_buy > 0:
            orders.append(Order("RAINFOREST_RESIN", bid, min(size, max_buy)))
        if max_sell > 0:
            orders.append(Order("RAINFOREST_RESIN", ask, -min(size, max_sell)))

        return orders

    def trade_kelp(self, mid_price, position, product):
        orders = []
        window = 10
        size = 5

        if len(self.history[product]) >= window:
            avg = np.mean(self.history[product][-window:])
            spread = 2
            # Round prices to integers
            bid = int(avg - spread / 2)
            ask = int(avg + spread / 2)

            max_buy = POSITION_LIMITS[product] - position
            max_sell = POSITION_LIMITS[product] + position

            if max_buy > 0:
                orders.append(Order(product, bid, min(size, max_buy)))
            if max_sell > 0:
                orders.append(Order(product, ask, -min(size, max_sell)))

        return orders

    def trade_squid_ink(self, mid_price, position, product):
        orders = []
        window = 20
        threshold = 1.0
        size = 5

        if len(self.history[product]) >= window:
            prices = self.history[product][-window:]
            mean = np.mean(prices)
            std = np.std(prices)
            z = (mid_price - mean) / std if std > 0 else 0

            max_buy = POSITION_LIMITS[product] - position
            max_sell = POSITION_LIMITS[product] + position

            # Use integer price for orders
            price = int(mid_price)

            if z < -threshold and max_buy > 0:
                orders.append(Order(product, price, min(size, max_buy)))
            elif z > threshold and max_sell > 0:
                orders.append(Order(product, price, -min(size, max_sell)))

        return orders