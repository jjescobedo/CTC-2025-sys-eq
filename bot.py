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
			# wait up to 0.5s for stdin readability so we don't busy-loop
			r, _, _ = select.select([sys.stdin], [], [], 0.5)
			if not r:
				continue

			# read the available line (this consumes the newline)
			line = sys.stdin.readline()
			if line == "":
				q.put("exit")
				break

			if line.strip() == "":
				pm = globals().get("print_manager")
				if pm is not None:
					pm.set_input_active(True)

				# show prompt using original_print (bypasses buffering)
				if pm is not None:
					pm.original_print("> ", end="", flush=True)
				else:
					builtins.print("> ", end="", flush=True)

				# read actual command (user types here)
				cmd = sys.stdin.readline()
				if cmd == "":
					if pm is not None:
						pm.set_input_active(False)
					q.put("exit")
					break

				cmd = cmd.strip()

				# turn off input buffering and flush buffered output
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
	# given
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
    Takes in the current fair values for every symbol and weighs the values then 
    returns the sum of the new weighted values.

    Input:
        fair_values: `dictionary` object w/ keys of type `str` and values of 
                     type `float` representing equity symbols and their values

    Output:
        fair_ETF: `float` object representing the current fair value of the ETF
    """
    fair_ETF = int()
    for symbol, value in fair_values.items():
        if symbol == "AAA":
            fair_ETF += value * globals()["unique_weight_aaa"]

        elif symbol == "BBB":
            fair_ETF += value * globals()["unique_weight_bbb"]
            
        elif symbol == "CCC":
            fair_ETF += value * globals()["unique_weight_ccc"]

    return fair_ETF

def check_ETF_discrepancy(expected_ETF_value: float, actual_ETF_value: float, spread: float) -> Union[tuple[str], None]:
    """
    Checks for significant discrepancy between expected_ETF_value and the actual_ETF_value,
    signalling for a long or short order on the ETF equity.

    Input:
        expected_ETF_value: `float` object representing the calculated, expected 
                            ETF equity value

        actual_ETF_value: `float` object representing the actual, reported ETF equity value

        spread: user-inputted `float` object representing threshold for 
                significant discrepancy
    
    Output:
        discrepancy: `tuple` object with single `str` object or type `None` representing
                     either a signal and, if so, what type of signal or no signal
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
	# initialize and activate print manager for the duration of the loop
	global print_manager
	print_manager = PrintManager()
	builtins.print = print_manager.print

	globals()["unique_weight_aaa"] = globals().get("unique_weight_aaa", 1.0)
	globals()["unique_weight_bbb"] = globals().get("unique_weight_bbb", 1.0)
	globals()["unique_weight_ccc"] = globals().get("unique_weight_ccc", 1.0)

	fair = generate_fair_values(symbols)
	print("Initial fair values:", fair)

	cmd_q: "queue.Queue[str]" = queue.Queue()
	stop_event = threading.Event()
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

	globals()["unique_weight"] = globals().get("unique_weight", 1.0)

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
									globals()["unique_weight_aaa"] = v

								elif sym == "BBB":
									globals()["unique_weight_bbb"] = v

								elif sym == "CCC":
									globals()["unique_weight_ccc"] = v

								else:
									print(f"Ignoring unknown symbol: {sym}")

							print("Weights:", {
								"AAA": globals()["unique_weight_aaa"],
								"BBB": globals()["unique_weight_bbb"],
								"CCC": globals()["unique_weight_ccc"],
							})

						elif arg.lower() == "all":
							if len(parts) < 3:
								print("Usage: setweight all VALUE")

							else:
								v = float(parts[2])
								globals()["unique_weight_aaa"] = globals()["unique_weight_bbb"] = globals()["unique_weight_ccc"] = v
								print(f"Set all weights = {v}")
								
						elif len(parts) >= 3 and parts[1].isalpha():
							sym = parts[1].upper()
							v = float(parts[2])

							if sym == "AAA":
								globals()["unique_weight_aaa"] = v

							elif sym == "BBB":
								globals()["unique_weight_bbb"] = v

							elif sym == "CCC":
								globals()["unique_weight_ccc"] = v

							else:
								print(f"Unknown symbol: {sym}")

							print(f"Set {sym} weight = {v}")

						else:
							v = float(arg)
							globals()["unique_weight_aaa"] = globals()["unique_weight_bbb"] = globals()["unique_weight_ccc"] = v
							print(f"Set all weights = {v}")

					except Exception as e:
						print(f"[ERR] setweight failed: {e}")

		except Exception as e:
			print(f"[ERR] Command failed: {e}")

		return True

	try:
		while True:
			# drain queue
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

			# normal market-making work
			update_fair_values(fair, drift_std=0.3)

			if check_ETF_discrepancy():
				pass

			for sym in symbols:
				fair_value = fair[sym]
				orders = make_bid_ask_orders(sym, fair_value)
                        
				for o in orders:
					print(f"current order: {o}")
					created = place_order(api_url, api_key, o)
                              
					if created:
						order_id = created.get("order_id") or created.get("id")
						if order_id:
							submit_net(cancel_order, api_url, api_key, order_id)
						else:
							print("[ERR] place_order returned no order_id, cannot cancel")
                                          
					else:
						print("[ERR] place_order failed, skipping cancel")

				print(f"[{sym}] fair={fair_value:.2f}\n")
				time.sleep(0.5)

			if not loop:
				break

			time.sleep(1)

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
