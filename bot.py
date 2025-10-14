#!/usr/bin/env python3
import os
import time
import json
import argparse
import random
from typing import Any, Optional, Union
import requests

import threading

# ----------------------------
# Helpers
# ----------------------------
def build_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key, "Content-Type": "application/json"}

def _raise_for_api_error(resp: requests.Response) -> None:
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
    url = f"{base_url}{path}"
    resp = requests.get(url, headers=build_headers(api_key), params=params, timeout=15)
    _raise_for_api_error(resp)
    return resp.json()


def api_post(base_url: str, path: str, api_key: str, body: dict[str, Any]) -> Any:
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
    """Send a single order to the API."""
    try:
        res = api_post(api_url, "/api/v1/orders", api_key, order)
        print(
            f"[OK] {order['side'].upper():4} {order['symbol']:5} "
            f"{order['quantity']:>3} @ {order.get('price', 'MKT')} ({order['order_type']})"
        )
        return res
    except Exception as e:
        print(f"[ERR] Failed order for {order['symbol']}: {e}")
        return None

def list_symbols(base_url: str, api_key: str) -> None:
    data = api_get(base_url, "/api/v1/symbols", api_key)
    symbols = data.get("symbols", [])
    if not symbols:
        print("No symbols available.")
        return
    print("Available symbols:")
    for row in symbols:
        print(f"  - {row.get('symbol')}\t{row.get('name')}")

def list_open_orders(base_url: str, api_key: str, symbol: Optional[str] = None) -> None:
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

def cancel_order(base_url: str, api_key: str, order: dict[str, Any]) -> None:
    """
    Input:
        base_url: `str` object representing the base API url

        api_key: `str` object representing the API key

        order: `dict` object representing a single order's data

    Output:
        None

    Cancel a single order.
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
    Input:
        base_url: `str` object representing the base API url

        api_key: `str` object representing the API key

        symbols: `list` object containing `str` objects representing the symbols to cancel all orders for 

    Output:
        None
    
    Cancel all orders across all equities or all orders for a single equity symbol.
    """
    # priv cancel function for readability sake
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

# ----------------------------
# Market-making logic
# ----------------------------
def generate_fair_values(symbols: list[str]) -> dict[str, float]:
    ### Needs to be updated
    fair = {}
    for s in symbols:
        fair[s] = round(random.uniform(90, 250), 2)
    return fair


def update_fair_values(fair: dict[str, float], drift_std: float = 0.5) -> None:
    ### Needs to be updated
    for s in fair:
        fair[s] += random.gauss(0, drift_std)
        fair[s] = round(fair[s], 2)

def make_bid_ask_orders(symbol: str, fair_value: float) -> list[dict[str, Any]]:
    spread = random.uniform(0.1, 0.6)
    qty = random.randint(25, 50)

    bid_px = round(fair_value - spread / 2, 2)
    ask_px = round(fair_value + spread / 2, 2)

    return [
        {"symbol": symbol, "side": "buy", "order_type": "limit", "quantity": qty, "price": bid_px},
        {"symbol": symbol, "side": "sell", "order_type": "limit", "quantity": qty, "price": ask_px},
    ]

def update_ETF_fair(fair_values: dict[str, float]) -> float:
    """
    Input:
        fair_values: `dictionary` object w/ keys of type `str` and values of 
                     type `float` representing equity symbols and their values

    Output:
        fair_ETF: `float` object representing the current fair value of the ETF

    Takes in the current fair values for every symbol and weighs the values then 
    returns the sum of the new weighted values.
    """
    fair_ETF = int()
    for symbol, value in fair_values.items():
        if symbol == "AAA":
            fair_ETF += value * unique_weight

        elif symbol == "BBB":
            fair_ETF += value * unique_weight
            
        elif symbol == "CCC":
            fair_ETF += value * unique_weight

    return fair_ETF

def check_ETF_discrepancy(expected_ETF_value: float, actual_ETF_value: float, spread: float) -> Union[tuple[str], None]:
    """
    Input:
        expected_ETF_value: `float` object representing the calculated, expected 
                            ETF equity value

        actual_ETF_value: `float` object representing the actual, reported ETF equity value

        spread: user-inputted `float` object representing threshold for 
                significant discrepancy
    
    Output:
        discrepancy: `tuple` object with single `str` object or type `None` representing
                     either a signal and, if so, what type of signal or no signal
    
    Checks for significant discrepancy between expected_ETF_value and the actual_ETF_value,
    signalling for a long or short order on the ETF equity.
    """
    diff = expected_ETF_value - actual_ETF_value
    if abs(diff) > spread:
        if diff > 0:
            return ("long",)
        elif diff < 0:
            return ("short",)
    else:
        return None

# ----------------------------
# Main trading loop
# ----------------------------
def market_making_loop(api_url: str, api_key: str, symbols: list[str], loop: bool = True):
    fair = generate_fair_values(symbols)
    print("Initial fair values:", fair)

    while True:
        # figure a way to queue terminal input?
        update_fair_values(fair, drift_std=0.3)
        # if (discrepancy := check_ETF_discrepancy()):
        #   if discrepancy[0] == "long":
        #       execute_order()
        #   else:
        #       execute_different_order()
        for sym in symbols:
            fair_value = fair[sym]
            orders = make_bid_ask_orders(sym, fair_value)
            for o in orders:
                print(f"current order: {o}")
                created = place_order(api_url, api_key, o)
                cancel_order(api_url, api_key, created.get("order_id"))
            print(f"[{sym}] fair={fair_value:.2f}\n")
            time.sleep(0.5)

        if not loop:
            break

        time.sleep(1)


# ----------------------------
# Entry point
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Automated Market Maker for CTC API")
    parser.add_argument("--api-url", default=os.environ.get("CTC_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("CTC_API_KEY") or os.environ.get("X_API_KEY"))
    parser.add_argument("--symbols", default="AAA,BBB,CCC,ETF", help="Comma-separated list of symbols")
    parser.add_argument("--loop", action="store_true", help="Continuously place orders")
    return parser.parse_args()


def main():
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
