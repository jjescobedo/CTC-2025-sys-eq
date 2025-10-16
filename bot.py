#!/usr/bin/env python3
import os
import time
import json
import argparse
import random
from typing import Any, Optional, Union, List
import requests

import threading
import queue
import concurrent.futures
import builtins
import sys
import select
import numpy as np
from dotenv import load_dotenv

API_URL_GLOBAL = ""
API_KEY_GLOBAL = ""
price_history = {"AAA": [], "BBB": [], "CCC": [], "ETF": []}
weights = {"AAA": [], "BBB": [], "CCC": []}
min_samples_for_regression = 50
max_etf_position = 75
base_spread = 0.02
# ----------------------------
# Helpers
# ----------------------------
def build_headers(api_key: str) -> dict[str, str]:
    # given
    return {"X-API-Key": api_key, "Content-Type": "application/json"}

def _raise_for_api_error(resp: requests.Response) -> None:
    # given
    if 200 <= resp.status_code < 300:
        return
    try:
        data = resp.json()
        detail = data.get("detail") if isinstance(data, dict) else None
    except Exception:
        detail = None
    msg = f"HTTP {resp.status_code}"
    if detail:
        msg += f": {detail}"
    raise RuntimeError(msg)


def api_get(base_url: str, path: str, api_key: str, params: Optional[dict[str, Any]] = None) -> Any:
    # given 
    url = f"{base_url}{path}"
    resp = requests.get(url, headers=build_headers(api_key), params=params, timeout=15)
    _raise_for_api_error(resp)
    return resp.json()


def api_post(base_url: str, path: str, api_key: str, body: dict[str, Any]) -> Any:
    # given
    url = f"{base_url}{path}"
    resp = requests.post(url, headers=build_headers(api_key), data=json.dumps(body), timeout=10)
    if not (200 <= resp.status_code < 300):
        try:
            err = resp.json().get("detail", "")
        except Exception:
            err = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {err}")
    return resp.json()


def place_order(api_url: str, api_key: str, order: dict[str, Any]) -> Optional[dict[str, Any]]:
    # given
    """Send a single order to the API."""
    try:
        res = api_post(api_url, "/api/v1/orders", api_key, order)
        print(
            f"[OK] {order['side'].upper():4} {order['symbol']:5} "
            f"{order['quantity']:>3} @ {order.get('price', 'MKT')} ({order['order_type']})"
        )
        return res # added dis to get order id
    except Exception as e:
        print(f"[ERR] Failed order for {order['symbol']}: {e}")
        return None

def list_symbols(base_url: str, api_key: str) -> None:
    # given
    data = api_get(base_url, "/api/v1/symbols", api_key)
    symbols = data.get("symbols", [])
    if not symbols:
        print("No symbols available.")
        return
    print("Available symbols:")
    for row in symbols:
        print(f"  - {row.get('symbol')}\t{row.get('name')}")

def list_open_orders(base_url: str, api_key: str, symbol: Optional[str] = None) -> None:
    # given
    params: dict[str, Any] = {}
    if symbol:
        params["symbol"] = symbol
    data = api_get(base_url, "/api/v1/orders/open", api_key, params=params)
    orders = data.get("orders", [])
    if not orders:
        print("No open orders.")
        return
    print("Open orders:")
    for o in orders:
        price_str = f" @ {o['price']}" if o.get("price") is not None else ""
        print(
            f"  - {o['order_id']} | {o['symbol']} {o['side']} {o['quantity']} {o['order_type']}{price_str} | {o['status']}"
        )

def get_order_book(base_url: str, api_key: str, symbol: str) -> Optional[dict[str, Any]]:
    """
    Fetches the order book for a given symbol.
    """
    try:
        return api_get(base_url, f"/api/v1/orderbook/{symbol}", api_key)
    
    except Exception as e:
        print(f"[ERR] Failed to get order book for {symbol}: {e}")
        return None

def get_market_trades(base_url: str, api_key: str, symbol: str, limit: int = 20) -> Optional[dict[str, Any]]:
    """
    Fetches the last N market trades for a given symbol.
    """
    try:
        return api_get(base_url, "/api/v1/trades/market", api_key, params={"symbol": symbol, "limit": limit})
    
    except Exception as e:
        print(f"[ERR] Failed to get market trades for {symbol}: {e}")
        return None

def get_mid_price(order_book: dict[str, Any]) -> Optional[float]:
    """
    Calculates the mid-price from an order book.
    """
    if not order_book or not order_book.get("bids") or not order_book.get("asks"):
        return None
    
    try:
        best_bid = float(order_book["bids"][0]["price"])
        best_ask = float(order_book["asks"][0]["price"])
        return round((best_bid + best_ask) / 2, 2)
    
    except (IndexError, KeyError, ValueError):
        return None

def get_volatility(trades: dict[str, Any]) -> float:
    """
    Calculates the standard deviation of recent trade prices.
    """
    if not trades or not trades.get("trades"):
        return 0.2
        
    prices = [float(t["price"]) for t in trades["trades"]]
    if len(prices) < 2:
        return 0.2
        
    vol = np.std(prices)
    return round(max(vol, 0.2), 2)

def get_positions(base_url: str, api_key: str) -> dict[str, float]:
    try:
        data = api_get(base_url, "/api/v1/positions", api_key)
        positions = data.get("positions", [])
        
        return {p["symbol"]: float(p["quantity"]) for p in positions}
    
    except Exception as e:
        print(f"[ERR] Failed to get positions: {e}", file=sys.stderr, flush=True)
        return {}

def cancel_order(base_url: str, api_key: str, order: dict[str, Any]) -> None:
    """
    Cancel a single order.
    
    Input:
        base_url: `str` object representing the base API url

        api_key: `str` object representing the API key

        order: `dict` object representing a single order's data

    Output:
        None
    """
    # normalize order_id
    if isinstance(order, (str, int)):
        order_id = str(order)

    elif isinstance(order, dict):
        order_id = str(order.get("order_id") or order.get("id") or "")

    else:
        print("[ERR] Invalid order parameter")
        return

    if not order_id:
        print("[ERR] Couldn't determine order_id")
        return

    url = f"{base_url}/api/v1/orders/{order_id}"
    try:
        resp = requests.delete(url, headers=build_headers(api_key), timeout=10)
        _raise_for_api_error(resp)
        print(f"[OK] Canceled {order_id}")

    except Exception as e:
        print(f"[ERR] Failed to cancel {order_id}, {e}")

def cancel_all_orders(base_url: str, api_key: str, symbols: list[str] = None) -> None:
    """
    Cancel all orders across all equities or all orders for a single equity symbol.
    
    Input:
        base_url: `str` object representing the base API url

        api_key: `str` object representing the API key

        symbols: `list` object containing `str` objects representing the symbols to cancel all orders for 

    Output:
        None
    """
    def _cancel_list(orders: list[dict[str, Any]]) -> None:
        if not orders:
            return
        
        for order in orders:
            cancel_order(base_url, api_key, order)
            time.sleep(0.05)

    if symbols: # iterate thru symbols to cancel if given
        for symbol in symbols:
            try:
                data = api_get(base_url, "/api/v1/orders/open", api_key, params={"symbol": symbol})

            except Exception as e:
                print(f"[ERR] Failed to list open orders for {symbol}, {e}")
                continue

            _cancel_list(data.get("orders", []))
    else:
        try:
            data = api_get(base_url, "/api/v1/orders/open", api_key)

        except Exception as e:
            print(f"[ERR] Failed to list open orders, {e}")
            return
        
        _cancel_list(data.get("orders", []))

def input_worker(q: "queue.Queue[str]", stop_event: "threading.Event") -> None:
    """
    Waits for stdin activity. If user presses enter on an empty line -> open an explicit prompt which
    enables print buffering, so background prints don't disrupt typing.
    
    If user types a non-empty line and hits enter, treat it directly as a command.
    """
    while not stop_event.is_set():
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0.5)
            if not r:
                continue

            line = sys.stdin.readline()
            if line == "":
                q.put("exit")
                break

            if line.strip() == "":
                pm = globals().get("print_manager")
                if pm is not None:
                    pm.set_input_active(True)

                if pm is not None:
                    pm.original_print("> ", end="", flush=True)
                else:
                    builtins.print("> ", end="", flush=True)

                cmd = sys.stdin.readline()
                if cmd == "":
                    if pm is not None:
                        pm.set_input_active(False)
                    q.put("exit")
                    break

                cmd = cmd.strip()

                if pm is not None:
                    pm.set_input_active(False)

                if cmd:
                    q.put(cmd)

            else:
                q.put(line.strip())

        except Exception:
            # ensure buffering is disabled on unexpected error and keep loop alive
            pm = globals().get("print_manager")
            if pm is not None:
                pm.set_input_active(False)
            continue

# ----------------------------
# Market-making logic
# ----------------------------
def generate_fair_values() -> dict[str, float]:
    fair = {
        "AAA": 50,
        "BBB": 25,
        "CCC": 75
    }
        
    return fair

def update_fair_values(fair: dict[str, float]) -> None:
    global API_URL_GLOBAL, API_KEY_GLOBAL
    
    for s in fair:
        book = get_order_book(API_URL_GLOBAL, API_KEY_GLOBAL, s)
        mid = get_mid_price(book)
        if mid:
            fair[s] = mid

def make_bid_ask_orders(symbol: str, fair_value: float) -> list[dict[str, Any]]:
    global API_URL_GLOBAL, API_KEY_GLOBAL, base_spread
    
    trades = get_market_trades(API_URL_GLOBAL, API_KEY_GLOBAL, symbol)
    volatility = get_volatility(trades)
    
    spread = round(base_spread + volatility * 0.2, 2) 
    qty = random.randint(25, 50)

    bid_px = round(fair_value - spread / 2, 2)
    ask_px = round(fair_value + spread / 2, 2)

    return [
        {"symbol": symbol, "side": "buy", "order_type": "limit", "quantity": qty, "price": bid_px},
        {"symbol": symbol, "side": "sell", "order_type": "limit", "quantity": qty, "price": ask_px},
    ]

def etf_action(base_url: str, api_key: str, etf_symbol: str, action: str, qty: int) -> Optional[dict[str, Any]]:
    """Helper to call the create or redeem ETF endpoint."""
    path = f"/api/v1/etf/{etf_symbol}/{action}"
    body = {"quantity": qty}
    try:
        res = api_post(base_url, path, api_key, body)
        print(f"[ARB] ETF {action} successful for {qty} units.")
        return res
    except Exception as e:
        print(f"[ERR] ETF {action} failed for {etf_symbol}: {e}", file=sys.stderr, flush=True)
        return None
    
def update_ETF_fair(fair_values: dict[str, float]) -> float:
    """
    Takes in the current fair values for every symbol and weighs the values then 
    returns the sum of the new weighted values.

    Input:
        fair_values: `dictionary` object w/ keys of type `str` and values of 
                     type `float` representing equity symbols and their values

    Output:
        fair_ETF: `float` object representing the current fair value of the ETF
    """
    global weights

    fair_ETF = int()
    for symbol, value in fair_values.items():
        if symbol == "AAA":
            fair_ETF += value * weights["AAA"]

        elif symbol == "BBB":
            fair_ETF += value * weights["BBB"]
            
        elif symbol == "CCC":
            fair_ETF += value * weights["CCC"]

    return fair_ETF

def check_and_execute_etf_arbitrage(expected_ETF_value: float, submit_net, positions: dict[str, float], profit_threshold: float = 0.10):
    global API_URL_GLOBAL, API_KEY_GLOBAL, max_etf_position
    
    book = get_order_book(API_URL_GLOBAL, API_KEY_GLOBAL, "ETF")
    if not book or not book.get("bids") or not book.get("asks"):
        print("[ARB] No ETF order book, skipping arb check.")
        return

    try:
        etf_market_bid = float(book["bids"][0]["price"])
        etf_market_ask = float(book["asks"][0]["price"])

    except Exception:
        print("[ARB] Incomplete ETF book, skipping arb check.")
        return

    current_etf_pos = positions.get("ETF", 0.0)
    profit_redeem = expected_ETF_value - etf_market_ask
    if profit_redeem > profit_threshold and current_etf_pos < max_etf_position:
        qty = 5

        qty_to_cap = max_etf_position - current_etf_pos
        qty = int(min(qty, qty_to_cap))

        if qty <= 0:
            return
        
        print(f"[ARB] REDEEM opportunity! Pos: {current_etf_pos} < {max_etf_position}. Buying {qty} units.")
        print(f"      Implied: ${expected_ETF_value:.2f}, Mkt Ask: ${etf_market_ask:.2f}. Profit: ${profit_redeem:.2f}")
        submit_net(etf_action, API_URL_GLOBAL, API_KEY_GLOBAL, "ETF", "redeem", qty)
        return

    elif profit_redeem > profit_threshold:
        print(f"[INFO] Skipping REDEEM (BUY) opportunity. Position {current_etf_pos} at/over cap {max_etf_position}.")

    profit_create = etf_market_bid - expected_ETF_value
    if profit_create > profit_threshold and current_etf_pos > -max_etf_position:
        qty = 5

        qty_to_cap = current_etf_pos - (-max_etf_position)
        qty = int(min(qty, qty_to_cap))

        if qty <= 0:
            return
        
        print(f"[ARB] CREATE opportunity! Pos: {current_etf_pos} > {-max_etf_position}. Selling {qty} units.")
        print(f"      Implied: ${expected_ETF_value:.2f}, Mkt Bid: ${etf_market_bid:.2f}. Profit: ${profit_create:.2f}")
        submit_net(etf_action, API_URL_GLOBAL, API_KEY_GLOBAL, "ETF", "create", qty)
        return
    
    elif profit_create > profit_threshold:
        print(f"[INFO] Skipping CREATE (SELL) opportunity. Position {current_etf_pos} at/under cap {-max_etf_position}.")

def calculate_etf_weights() -> bool:
    global price_history, min_samples_for_regression, weights
    print("[INFO] Attempting to calculate ETF weights...")
    
    try:
        x_matrix = np.array([
            price_history["AAA"],
            price_history["BBB"],
            price_history["CCC"]
        ]).T

        y_vector = np.array(price_history["ETF"])

        results = np.linalg.lstsq(x_matrix, y_vector, rcond=None)
        
        coefficients = results[0]
        
        weights["AAA"] = coefficients[0]
        weights["BBB"] = coefficients[1]
        weights["CCC"] = coefficients[2]
        
        print(f"[SUCCESS] Calculated ETF weights:")
        print(f"  - AAA weight: {coefficients[0]:.4f}")
        print(f"  - BBB weight: {coefficients[1]:.4f}")
        print(f"  - CCC weight: {coefficients[2]:.4f}")
        
        price_history = {k: [] for k in price_history}
        return True

    except Exception as e:
        print(f"[ERR] Failed to calculate ETF weights: {e}")
        
        weights["AAA"] = 1.0
        weights["BBB"] = 1.0
        weights["CCC"] = 1.0
        return False

# ----------------------------
# Main trading loop
# ----------------------------
def market_making_loop(api_url: str, api_key: str, symbols: list[str], loop: bool = True):
    global print_manager
    print_manager = PrintManager()
    builtins.print = print_manager.print

    global API_URL_GLOBAL, API_KEY_GLOBAL
    global price_history, min_samples_for_regression, weights
    API_URL_GLOBAL = api_url
    API_KEY_GLOBAL = api_key

    weights["AAA"] = 1.0
    weights["BBB"] = 1.0
    weights["CCC"] = 1.0

    fair = generate_fair_values()
    print("Initial fair values:", fair)

    cmd_q: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    input_thread = threading.Thread(target=input_worker, args=(cmd_q, stop_event), daemon=True)
    input_thread.start()
    
    def submit_net(fn, *a, **kw):
        return executor.submit(fn, *a, **kw)

    def handle_command(line: str) -> bool:
        """
        Commands:
          - exit
          - list
          - list_open [SYMBOL]
          - cancel ORDER_ID
          - cancel_all [SYMBOL]
          - place SYMBOL SIDE QTY [PRICE]
          - setweight SYMBOL=VAL[,SYMBOL=VAL...]
          - adjustspread -- | ++ [AMOUNT]
          - setmaxetfpos VALUE
        """
        parts = line.split()
        cmd = parts[0].lower()
        try:
            if cmd == "exit":
                print("exiting on user request")
                return False
            
            elif cmd == "list":
                submit_net(list_symbols, api_url, api_key)

            elif cmd == "list_open":
                sym = parts[1].upper() if len(parts) > 1 else None
                submit_net(list_open_orders, api_url, api_key, sym)

            elif cmd == "cancel":
                if len(parts) < 2:
                    print("Usage: cancel ORDER_ID")

                else:
                    order_id = parts[1]
                    submit_net(cancel_order, api_url, api_key, order_id)
            
            elif cmd == "cancel_all":
                if len(parts) > 1:
                    syms = [s.strip().upper() for s in parts[1].split(",") if s.strip()]
                    submit_net(cancel_all_orders, api_url, api_key, syms)
                
                else:
                    submit_net(cancel_all_orders, api_url, api_key, None)

            elif cmd == "place":
                if len(parts) < 4:
                    print("Usage: place SYMBOL SIDE QTY [PRICE]")
                
                else:
                    sym = parts[1].upper()
                    side = parts[2].lower()
                    qty = int(parts[3])
                    px = float(parts[4]) if len(parts) > 4 else None
                    order = {
                        "symbol": sym,
                        "side": side,
                        "order_type": "limit" if px is not None else "market",
                        "quantity": qty,
                    }
                    
                    if px is not None:
                        order["price"] = px

                    submit_net(place_order, api_url, api_key, order)
            
            elif cmd == "setweight":
                if len(parts) < 2:
                    print("Usage: setweight SYMBOL=VAL[,SYMBOL=VAL...] | setweight all VALUE")

                else:
                    try:

                        arg = parts[1]
                        if "=" in arg or ("," in " ".join(parts[1:])):
                            s = " ".join(parts[1:])
                            pairs = [p.strip() for p in s.split(",") if p.strip()]

                            for p in pairs:
                                if "=" not in p:
                                    raise ValueError(f"bad pair: {p}")
                                
                                sym, val = p.split("=", 1)
                                sym = sym.strip().upper()
                                v = float(val)
                                if sym == "AAA":
                                    weights["AAA"] = v

                                elif sym == "BBB":
                                    weights["BBB"] = v

                                elif sym == "CCC":
                                    weights["CCC"] = v

                                else:
                                    print(f"Ignoring unknown symbol: {sym}")

                            print("Weights:", {
                                "AAA": weights["AAA"],
                                "BBB": weights["BBB"],
                                "CCC": weights["CCC"],
                            })

                        elif arg.lower() == "all":
                            if len(parts) < 3:
                                print("Usage: setweight all VALUE")

                            else:
                                v = float(parts[2])
                                weights["AAA"] = weights["BBB"] = weights["CCC"] = v
                                print(f"Set all weights = {v}")
                                
                        elif len(parts) >= 3 and parts[1].isalpha():
                            sym = parts[1].upper()
                            v = float(parts[2])

                            if sym == "AAA":
                                weights["AAA"] = v

                            elif sym == "BBB":
                                weights["BBB"] = v

                            elif sym == "CCC":
                                weights["CCC"] = v

                            else:
                                print(f"Unknown symbol: {sym}")

                            print(f"Set {sym} weight = {v}")

                        else:
                            v = float(arg)
                            weights["AAA"] = weights["BBB"] = weights["CCC"] = v
                            print(f"Set all weights = {v}")

                    except Exception as e:
                        print(f"[ERR] setweight failed: {e}")
            
            elif cmd == "adjustspread":
                if len(parts) < 2:
                    print("Usage: adjustspread -- | ++ [AMOUNT]")
                
                else:
                    try:
                        global base_spread
                        arg = parts[1]
                        change_amount = int()
                        try:
                            change_amount = int(parts[2])

                        except:
                            print("[OK] No change amount given, one adjustment made")
                            change_amount = 1

                        adjustment_amount = 0.005

                        if arg == "--":
                            for _ in range(change_amount):
                                base_spread = round(max(0.01, base_spread - adjustment_amount), 3)
                            print(f"[OK] Base spread lowered by {adjustment_amount * change_amount}, now {base_spread}")
                        
                        elif arg == "++":
                            for _ in range(change_amount):
                                base_spread = round(base_spread + adjustment_amount, 3)
                            print(f"[OK] Base spread increased by {adjustment_amount * change_amount}, now {base_spread}")
                        
                        else:
                            print("[ERR] Unknown argument. Use -- (tighter) or ++ (wider)")

                    except Exception as e:
                        print(f"[ERR] Failed to adjust spread: {e}")
                
            elif cmd == "setmaxetfpos":
                if len(parts) < 2:
                    print("Usage: setmaxetfpos VALUE")
                
                else:
                    try:
                        global max_etf_position
                        arg = parts[1]
                        max_etf_position = int(arg)
                        print(f"[OK] Max ETF position set to {arg}")
                    
                    except Exception as e:
                        print(f"[ERR] Failed to set max etf position: {e}")

        except Exception as e:
            print(f"[ERR] Command failed: {e}")

        return True

    weights_calculated = False
    all_symbols_with_etf = list(set(symbols + ["ETF"]))
    component_symbols = [symbol for symbol in symbols if symbol != "ETF"]

    try:
        while True:
            #list_open_orders(api_url, api_key)
            #list_symbols(api_url, api_key)
            
            while True:
                try:
                    line = cmd_q.get_nowait()

                except queue.Empty:
                    break

                cont = handle_command(line)
                if not cont:
                    stop_event.set()
                    executor.shutdown(wait=True)
                    return
                
            if not weights_calculated:
                current_mids = dict()
                all_data_present = True

                for symbol in all_symbols_with_etf:
                    book = get_order_book(api_url, api_key, symbol)
                    mid = get_mid_price(book)
                
                    if mid:
                        current_mids[symbol] = mid
                    
                    else:
                        all_data_present = False
                        break
                
                if all_data_present:
                    for symbol in all_symbols_with_etf:
                        price_history[symbol].append(current_mids[symbol])
                
                if len(price_history["ETF"]) >= min_samples_for_regression:
                    weights_calculated = calculate_etf_weights()
                
                time.sleep(1)
                continue

            current_positions = get_positions(api_url, api_key)
            submit_net(cancel_all_orders, api_url, api_key, component_symbols)
            update_fair_values(fair)
            fair_ETF = update_ETF_fair(fair)
            submit_net(check_and_execute_etf_arbitrage, fair_ETF, submit_net, current_positions, profit_threshold=0.10)

            for symbol in component_symbols:
                fair_value = fair[symbol]
                orders = make_bid_ask_orders(symbol, fair_value)
                        
                for order in orders:
                    submit_net(place_order, api_url, api_key, order)

                print(f"[{symbol}] Quoting around fair={fair_value:.2f}\n")
                
            print(f"Implied ETF: {fair_ETF:.2f}, | Weights: A={weights["AAA"]:.2f}, B={weights["BBB"]:.2f}, C={weights["CCC"]:.2f}\n")

            if not loop:
                break

            time.sleep(5)

    finally:
        if print_manager is not None:
            builtins.print = print_manager.original_print

        stop_event.set()
        executor.shutdown(wait=False)
        time.sleep(0.1)

class PrintManager:
    """
    Buffers print output while input is active, then flushes after input completes.
    Replaces builtins.print during the market loop.
    """
    def __init__(self):
        self.original_print = builtins.print
        self.lock = threading.Lock()
        self.input_active = False
        self.buffer: List[str] = []

    def set_input_active(self, active: bool) -> None:
        with self.lock:
            # when turning input off, flush buffered lines
            self.input_active = active
            if not self.input_active and self.buffer:
                for line in self.buffer:
                    self.original_print(line)
                              
                self.buffer.clear()

    def print(self, *args, **kwargs) -> None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        flush = kwargs.get("flush", False)
        msg = sep.join(str(a) for a in args) + end

        with self.lock:
            if self.input_active:
                # buffer entire message (including trailing newline)
                self.buffer.append(msg.rstrip("\n"))
                return
                  
            # not in input, go straight to original print
            self.original_print(*args, sep=sep, end=end, flush=flush)

print_manager: PrintManager | None = None

# ----------------------------
# Entry point
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Automated Market Maker for CTC API")
    parser.add_argument("--api-url", default=os.environ.get("CTC_API_URL", "https://cornelltradingcompetition.org"))
    parser.add_argument("--api-key", default=os.environ.get("CTC_API_KEY") or os.environ.get("X_API_KEY"))
    parser.add_argument("--symbols", default="AAA,BBB,CCC,ETF", help="Comma-separated list of symbols")
    parser.add_argument("--loop", action="store_true", help="Continuously place orders")
    return parser.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    api_key = args.api_key or input("Enter API key: ").strip()
    if not api_key:
        print("API key required.")
        return 1

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    market_making_loop(args.api_url, api_key, symbols, loop=args.loop)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
