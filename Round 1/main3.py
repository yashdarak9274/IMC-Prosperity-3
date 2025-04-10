from typing import Dict, List
from datamodel import Order, TradingState 
import numpy as np
import json

class Trader:
    def __init__(self):
        self.history = {}
        self.position_limits = {}
        self.printed_products = False
        self.default_position_limit = 50
        self.default_spread = 2
        self.min_history_for_stats = 10

    def adapt_to_products(self, products):
        """Adapt strategy to available products"""
        for product in products:
            if product not in self.position_limits:
                self.position_limits[product] = self.default_position_limit

    def run(self, state: TradingState):
        result = {}

        # Print available products once
        if not self.printed_products:
            print(f"Available products in state: {list(state.order_depths.keys())}")
            self.printed_products = True
            # Set position limits for all products
            for product in state.order_depths:
                self.position_limits[product] = self.default_position_limit

        # Load historical data from traderData string
        if state.traderData:
            try:
                self.history = json.loads(state.traderData)
            except:
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
            spread = best_ask - best_bid

            position = state.position.get(product, 0)

            # Initialize price history list if not present
            if product not in self.history:
                self.history[product] = []
            self.history[product].append(mid_price)

            # Determine trading strategy based on product and history
            if len(self.history[product]) < self.min_history_for_stats:
                # Simple market making for products with limited history
                orders = self.market_make(product, mid_price, position, spread)
            else:
                # Choose strategy based on product characteristics
                volatility = np.std(self.history[product][-20:])
                
                if volatility > 5:
                    # High volatility: trend following
                    orders = self.trend_following(product, position)
                elif volatility < 2:
                    # Low volatility: mean reversion
                    orders = self.mean_reversion(product, position)
                else:
                    # Medium volatility: market making
                    orders = self.market_make(product, mid_price, position, spread)

            result[product] = orders

        # Properly persist the history as JSON
        traderData = json.dumps(self.history)

        return result, 0, traderData

    def market_make(self, product, mid_price, position, market_spread):
        """Basic market making strategy"""
        orders = []
        
        # Dynamic spread based on market conditions
        spread = max(self.default_spread, market_spread * 0.8)
        
        # Round prices to integers
        bid = int(mid_price - spread/2)
        ask = int(mid_price + spread/2)

        # Position limit for this product
        limit = self.position_limits.get(product, self.default_position_limit)
