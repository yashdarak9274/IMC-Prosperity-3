import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datamodel import TradingState, OrderDepth, Order, Listing, Observation, Trade
import json
import importlib.util
import sys
from datetime import datetime
import os

class BacktestEngine:
    def __init__(self, data_files, strategy_files):
        self.data_files = data_files
        self.strategy_files = strategy_files
        self.strategies = {}
        self.results = {}
        self.load_strategies()
        
    def load_strategies(self):
        """Load trading strategies from files"""
        for i, file_path in enumerate(self.strategy_files):
            strategy_name = f"strategy_{i+1}"
            spec = importlib.util.spec_from_file_location(strategy_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[strategy_name] = module
            spec.loader.exec_module(module)
            self.strategies[strategy_name] = module.Trader()
            self.results[strategy_name] = {
                "pnl": [], 
                "positions": {}, 
                "trades": [],
                "daily_pnl": {}
            }
    
    def inspect_data_files(self):
        """Inspect the structure of data files"""
        for file_path in self.data_files:
            print(f"\nInspecting {file_path}...")
            df = pd.read_csv(file_path)
            
            print(f"Shape: {df.shape}")
            print(f"Columns (first 10): {df.columns.tolist()[:10]}...")
            
            # Find product names
            products = set()
            for col in df.columns:
                if 'bidprice' in col.lower():
                    product = col.lower().split('bidprice')[0]
                    products.add(product)
            
            print(f"Products found: {products}")
            
            # Sample data for first product
            if products:
                product = list(products)[0]
                print(f"\nSample data for {product}:")
                relevant_cols = [col for col in df.columns if product in col.lower()]
                print(df[relevant_cols].head())
            
            return products  # Return products for strategy adaptation
    
    def load_market_data(self, file_path):
        """Load historical market data"""
        df = pd.read_csv(file_path)
        
        # Convert column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Parse timestamp if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(int)
        else:
            df['timestamp'] = df.index
        
        return df
    
    def create_trading_state(self, timestamp, data_row, positions, trader_data=""):
        """Create TradingState object from data"""
        listings = {}
        order_depths = {}
        
        # Extract product names from column headers
        products = set()
        for column in data_row.index:
            if 'bidprice' in column:
                product = column.split('bidprice')[0]
                products.add(product)
        
        # Create order depths from data
        for product in products:
            order_depth = OrderDepth()
            has_orders = False
            
            # Populate with bids
            for i in range(1, 4):  # Assuming up to 3 levels
                bid_price_col = f"{product}bidprice{i}"
                bid_vol_col = f"{product}bidvolume{i}"
                
                if bid_price_col in data_row and bid_vol_col in data_row:
                    price = data_row[bid_price_col]
                    volume = data_row[bid_vol_col]
                    if not pd.isna(price) and not pd.isna(volume) and price > 0 and volume > 0:
                        order_depth.buy_orders[int(price)] = int(volume)
                        has_orders = True
            
            # Populate with asks
            for i in range(1, 4):  # Assuming up to 3 levels
                ask_price_col = f"{product}askprice{i}"
                ask_vol_col = f"{product}askvolume{i}"
                
                if ask_price_col in data_row and ask_vol_col in data_row:
                    price = data_row[ask_price_col]
                    volume = data_row[ask_vol_col]
                    if not pd.isna(price) and not pd.isna(volume) and price > 0 and volume > 0:
                        order_depth.sell_orders[int(price)] = int(volume)
                        has_orders = True
            
            # Only add products with valid orders
            if has_orders:
                order_depths[product] = order_depth
                listings[product] = Listing(product, product, "USD")
        
        # Create empty observations (or populate if available)
        observations = Observation({}, {})
        
        # Create TradingState
        state = TradingState(
            traderData=trader_data,
            timestamp=timestamp,
            listings=listings,
            order_depths=order_depths,
            own_trades={},
            market_trades={},
            position=positions,
            observations=observations
        )
        
        return state
    
    def calculate_pnl(self, trades, positions, current_data):
        """Calculate PnL for trades and positions"""
        trade_pnl = 0
        for trade in trades:
            if trade.quantity > 0:  # Buy trade
                trade_pnl -= trade.price * trade.quantity
            else:  # Sell trade
                trade_pnl += trade.price * abs(trade.quantity)
        
        position_pnl = 0
        for product, position in positions.items():
            # Try different column formats for mid price
            for mid_col in [f"{product}midprice", f"{product}mid", f"{product}price"]:
                if mid_col in current_data and not pd.isna(current_data[mid_col]):
                    position_pnl += position * current_data[mid_col]
                    break
            
            # If no mid price, use average of best bid and ask
            if position_pnl == 0:
                bid_col = f"{product}bidprice1"
                ask_col = f"{product}askprice1"
                if bid_col in current_data and ask_col in current_data:
                    if not pd.isna(current_data[bid_col]) and not pd.isna(current_data[ask_col]):
                        mid = (current_data[bid_col] + current_data[ask_col]) / 2
                        position_pnl += position * mid
        
        return trade_pnl + position_pnl
    
    def execute_orders(self, orders, market_data, product_positions):
        """Simulate order execution"""
        executed_trades = []
        
        for product, order_list in orders.items():
            for order in order_list:
                # For buy orders
                if order.quantity > 0:
                    # Try to find any ask price
                    for i in range(1, 4):
                        ask_price_col = f"{product}askprice{i}"
                        ask_vol_col = f"{product}askvolume{i}"
                        
                        if ask_price_col in market_data and ask_vol_col in market_data:
                            if not pd.isna(market_data[ask_price_col]) and not pd.isna(market_data[ask_vol_col]):
                                market_price = int(market_data[ask_price_col])
                                available_volume = int(market_data[ask_vol_col])
                                
                                # Execute if price is acceptable and volume is available
                                if order.price >= market_price and available_volume > 0:
                                    # Determine executed quantity
                                    exec_qty = min(order.quantity, available_volume)
                                    
                                    executed_trades.append(Trade(
                                        symbol=product,
                                        price=market_price,
                                        quantity=exec_qty,
                                        buyer="BACKTEST",
                                        seller="MARKET"
                                    ))
                                    
                                    product_positions[product] = product_positions.get(product, 0) + exec_qty
                                    break
                
                # For sell orders
                elif order.quantity < 0:
                    # Try to find any bid price
                    for i in range(1, 4):
                        bid_price_col = f"{product}bidprice{i}"
                        bid_vol_col = f"{product}bidvolume{i}"
                        
                        if bid_price_col in market_data and bid_vol_col in market_data:
                            if not pd.isna(market_data[bid_price_col]) and not pd.isna(market_data[bid_vol_col]):
                                market_price = int(market_data[bid_price_col])
                                available_volume = int(market_data[bid_vol_col])
                                
                                # Execute if price is acceptable and volume is available
                                if order.price <= market_price and available_volume > 0:
                                    # Determine executed quantity (negative for sells)
                                    exec_qty = max(order.quantity, -available_volume)
                                    
                                    executed_trades.append(Trade(
                                        symbol=product,
                                        price=market_price,
                                        quantity=exec_qty,
                                        buyer="MARKET",
                                        seller="BACKTEST"
                                    ))
                                    
                                    product_positions[product] = product_positions.get(product, 0) + exec_qty
                                    break
        
        return executed_trades
    
    def run_backtest(self):
        """Run backtest on historical data"""
        # Get products from first data file for strategy adaptation
        products = self.inspect_data_files()
        
        for file_path in self.data_files:
            print(f"\nProcessing {file_path}...")
            data = self.load_market_data(file_path)
            day = os.path.basename(file_path).split('_')[-1].split('.')[0]
            
            for strategy_name, strategy in self.strategies.items():
                positions = self.results[strategy_name]["positions"]
                trader_data = ""
                daily_pnl = 0
                daily_trades = 0
                
                for idx, row in data.iterrows():
                    timestamp = int(row.get('timestamp', idx))
                    
                    # Create trading state from data
                    state = self.create_trading_state(timestamp, row, positions, trader_data)
                    
                    # Adapt strategy to available products if needed
                    if hasattr(strategy, 'adapt_to_products') and products:
                        strategy.adapt_to_products(products)
                    
                    # Run strategy
                    orders, conversions, trader_data = strategy.run(state)
                    
                    # Process orders and update positions
                    trades = self.execute_orders(orders, row, positions)
                    daily_trades += len(trades)
                    self.results[strategy_name]["trades"].extend(trades)
                    
                    # Calculate PnL
                    pnl = self.calculate_pnl(trades, positions, row)
                    daily_pnl += pnl
                    self.results[strategy_name]["pnl"].append(pnl)
                
                # Store daily PnL
                self.results[strategy_name]["daily_pnl"][day] = daily_pnl
                print(f"{strategy_name} - Day {day}: PnL={daily_pnl:.2f}, Trades={daily_trades}")
        
        return self.results
    
    def plot_results(self):
        """Plot backtest results"""
        plt.figure(figsize=(15, 10))
        
        # Plot cumulative PnL
        plt.subplot(2, 1, 1)
        for strategy_name, result in self.results.items():
            cumulative_pnl = np.cumsum(result["pnl"])
            plt.plot(cumulative_pnl, label=strategy_name)
        
        plt.title("Cumulative PnL Comparison")
        plt.xlabel("Time Steps")
        plt.ylabel("Cumulative PnL")
        plt.legend()
        plt.grid(True)
        
        # Plot daily PnL
        plt.subplot(2, 1, 2)
        days = sorted(list(next(iter(self.results.values()))["daily_pnl"].keys()))
        
        x = np.arange(len(days))
        width = 0.35
        
        for i, (strategy_name, result) in enumerate(self.results.items()):
            daily_pnls = [result["daily_pnl"].get(day, 0) for day in days]
            plt.bar(x + i*width, daily_pnls, width, label=strategy_name)
        
        plt.title("Daily PnL Comparison")
        plt.xlabel("Day")
        plt.ylabel("PnL")
        plt.xticks(x + width/2, days)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("backtest_results.png")
        plt.show()

    def print_summary(self):
        """Print performance summary"""
        print("\n===== PERFORMANCE SUMMARY =====")
        
        for strategy_name, result in self.results.items():
            print(f"\n{strategy_name}:")
            
            # Calculate metrics
            cumulative_pnl = np.sum(result["pnl"])
            daily_pnls = list(result["daily_pnl"].values())
            
            # Calculate drawdown
            cumulative = np.cumsum(result["pnl"])
            max_dd = 0
            peak = cumulative[0] if len(cumulative) > 0 else 0
            
            for value in cumulative:
                if value > peak:
                    peak = value
                dd = (peak - value)
                if dd > max_dd:
                    max_dd = dd
            
            # Print metrics
            print(f"  Total PnL: {cumulative_pnl:.2f}")
            print(f"  Daily PnL: {daily_pnls}")
            print(f"  Max Drawdown: {max_dd:.2f}")
            
            # Print position summary
            print("  Final Positions:")
            for product, pos in result["positions"].items():
                print(f"    {product}: {pos}")
            
            # Print trade summary
            if result["trades"]:
                buy_trades = [t for t in result["trades"] if t.quantity > 0]
                sell_trades = [t for t in result["trades"] if t.quantity < 0]
                
                print(f"  Total Trades: {len(result['trades'])}")
                print(f"    Buy Trades: {len(buy_trades)}")
                print(f"    Sell Trades: {len(sell_trades)}")

# Run the backtest
if __name__ == "__main__":
    data_files = ["data/prices_round_1_day_-2.csv", "data/prices_round_1_day_-1.csv", "data/prices_round_1_day_0.csv"]
    strategy_files = ["main.py", "main2.py"]
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    backtest = BacktestEngine(data_files, strategy_files)
    results = backtest.run_backtest()
    backtest.plot_results()
    backtest.print_summary()
