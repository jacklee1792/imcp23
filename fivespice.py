import json
from typing import Callable, Optional

import numpy as np

from datamodel import Order, ProsperityEncoder, Symbol, Trade, TradingState

BANANAS = "BANANAS"
PEARLS = "PEARLS"
COCOS = "COCONUTS"
PINAS = "PINA_COLADAS"
GEAR = "DIVING_GEAR"
BERRIES = "BERRIES"

PICNIC_BASKET = "PICNIC_BASKET"
DIP = "DIP"
UKULELE = "UKULELE"
BAGUETTE = "BAGUETTE"

DOLPHINS = "DOLPHIN_SIGHTINGS"

POSITION_LIMITS = {
    BANANAS: 20,
    PEARLS: 20,
    COCOS: 600,
    PINAS: 300,
    GEAR: 50,
    BERRIES: 250,
    PICNIC_BASKET: 70,
    DIP: 300,
    UKULELE: 70,
    BAGUETTE: 150,
}

CAESAR = "Caesar"
CAMILLA = "Camilla"
CHARLIE = "Charlie"
GARY = "Gary"
GINA = "Gina"
OLGA = "Olga"
OLIVIA = "Olivia"
PABLO = "Pablo"
PARIS = "Paris"
PENELOPE = "Penelope"
PETER = "Peter"


class Logger:
    logs: str
    local: bool
    local_logs: dict[int, str]

    def __init__(self, local: bool = False) -> None:
        self.logs = ""
        self.local = local
        self.local_logs = dict()

    def print(self, *objects: any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        output = json.dumps(
            {
                "state": self.compress_state(state),
                "orders": self.compress_orders(orders),
                "logs": self.logs,
            },
            cls=ProsperityEncoder,
            separators=(",", ":"),
            sort_keys=True,
        )
        if self.local:
            self.local_logs[state.timestamp] = output
        else:
            print(output)
        self.logs = ""

    def compress_state(self, state: TradingState) -> dict[str, any]:
        listings = []
        for listing in state.listings.values():
            listing: any
            if self.local:
                items = [listing.symbol, listing.product, listing.denomination]
            else:
                items = [listing["symbol"], listing["product"], listing["denomination"]]
            listings.append(items)

        order_depths = {}
        for symbol, order_depth in state.order_depths.items():
            order_depths[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return {
            "t": state.timestamp,
            "l": listings,
            "od": order_depths,
            "ot": self.compress_trades(state.own_trades),
            "mt": self.compress_trades(state.market_trades),
            "p": state.position,
            "o": state.observations,
        }

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.buyer,
                        trade.seller,
                        trade.price,
                        trade.quantity,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed


class OrderManager:
    """
    Manages orders for a single symbol, ensuring that the position limit is never
    exceeded. Orders causing the position limit to be exceeded will have their
    quantity reduced so that the position limit is not exceeded.

    Also provides information about the current state of the market.
    """

    symbol: str
    state: TradingState
    position: int
    _orders: list[Order]

    # Buy quota is always positive, sell quota is always negative to be
    # consistent with order quantity formats.
    buy_quota: int
    sell_quota: int

    def __init__(self, symbol: str, state: TradingState):
        self.symbol = symbol
        self.state = state
        self.position = state.position.get(symbol, 0)
        self._orders = []
        self.buy_quota = POSITION_LIMITS[symbol] - self.position
        self.sell_quota = -POSITION_LIMITS[symbol] - self.position
        assert self.buy_quota >= 0

    def make_order(self, price: int, quantity: int):
        if quantity > 0:
            quantity = min(quantity, self.buy_quota)
            self.buy_quota -= quantity
        else:
            quantity = max(quantity, self.sell_quota)
            self.sell_quota -= quantity
        if quantity != 0:
            self._orders.append(Order(self.symbol, price, quantity))

    def make_buy_order(self, price: int, quantity: int, strict: bool = True):
        if quantity >= 0:
            self.make_order(price, quantity)
            return
        if strict:
            raise ValueError("Buy order quantity must be >= 0")

    def make_sell_order(self, price: int, quantity: int, strict: bool = True):
        if quantity <= 0:
            self.make_order(price, quantity)
            return
        if strict:
            raise ValueError("Sell order quantity must be <= 0")

    def pending_orders(self) -> list[Order]:
        return [order for order in self._orders if order.quantity != 0]

    def buy_orders(self) -> dict[int, int]:
        if (od := self.state.order_depths.get(self.symbol)) is None:
            return {}
        return od.buy_orders

    def sell_orders(self) -> dict[int, int]:
        if (od := self.state.order_depths.get(self.symbol)) is None:
            return {}
        return od.sell_orders

    def highest_bid(self) -> Optional[int]:
        buy_orders = self.buy_orders()
        return 0 if len(buy_orders) == 0 else max(buy_orders.keys())

    def lowest_ask(self) -> Optional[int]:
        sell_orders = self.sell_orders()
        return 0 if len(sell_orders) == 0 else min(sell_orders.keys())

    def mid_price(self) -> Optional[float]:
        buy_orders = self.buy_orders()
        sell_orders = self.sell_orders()
        total, volume = 0, 0
        for price, qty in buy_orders.items():
            total += price * qty
            volume += qty
        for price, qty in sell_orders.items():
            total += price * -qty
            volume += -qty
        return total / volume

    def position_adjustment(self, adjustments: list[int]):
        lim = POSITION_LIMITS[self.symbol]
        cutoffs = np.linspace(-lim, lim, len(adjustments) + 1)
        for adj, cutoff in zip(adjustments, cutoffs[1:-1]):
            if self.position <= cutoff:
                return adj
        return adjustments[-1]


class State:
    """
    State composes the TradingState class and provides order managers for each symbol.
    """

    state: TradingState
    order_managers: dict[str, OrderManager]

    def __init__(self, state: TradingState):
        self.state = state
        self.order_managers = dict()

    def order_manager(self, symbol: str) -> OrderManager:
        if symbol not in self.order_managers:
            mgr = OrderManager(symbol, self.state)
            self.order_managers[symbol] = mgr
        return self.order_managers[symbol]


class Parameter:
    params: dict[str, list[any]] = dict()
    tuning_index: Optional[dict[str, int]] = None

    name: str
    default: any

    def __init__(self, name: str, default: any):
        self.name = name
        self.default = default

    def get(self) -> any:
        p = Parameter.params.get(self.name)
        return p[self.tuning_index[self.name]] if p is not None else self.default


# ------- Traders begin here -----------------------------------------------------------


class PearlTrader:
    def run(self, s: State):
        mgr = s.order_manager(PEARLS)

        # Trade with all bids above 10000
        buy_orders = mgr.buy_orders()
        for bid in sorted(buy_orders.keys(), reverse=True):
            qty = buy_orders[bid]
            if bid > 10000:
                mgr.make_sell_order(bid, -qty)

        # Trade with all asks below 10000
        sell_orders = mgr.sell_orders()
        for ask in sorted(sell_orders.keys()):
            qty = sell_orders[ask]
            if ask < 10000:
                mgr.make_buy_order(ask, -qty)

        adj = mgr.position_adjustment(adjustments=[-2, -1, -1, 0, 1, 1, 2])
        bp = 9998 - adj
        sp = 10002 - adj
        buy_book = [1, 10, 10, 15]

        # Submit extra buy orders
        for i, qty in enumerate(buy_book):
            mgr.make_buy_order(bp - i, qty)
        mgr.make_buy_order(bp - len(buy_book), mgr.buy_quota)

        # Submit extra sell orders
        for i, qty in enumerate(buy_book):
            mgr.make_sell_order(sp + i, -qty)
        mgr.make_sell_order(sp + len(buy_book), mgr.sell_quota)


class CocoPinaTrader:
    def run(self, s: State):
        pc_mgr = s.order_manager(PINAS)
        c_mgr = s.order_manager(COCOS)

        c = Parameter("coconut_c", default=2).get()
        boundary = Parameter("coconut_boundary", default=20).get()
        exponent = Parameter("coconut_exponent", default=1).get()
        overall_spread = 15 / 8 * c_mgr.mid_price() - pc_mgr.mid_price()
        qty = round(c * (abs(overall_spread) ** exponent))
        logger.print(overall_spread, qty)

        if overall_spread > boundary:
            pc_mgr.make_buy_order(pc_mgr.lowest_ask(), qty)
            c_mgr.make_sell_order(c_mgr.highest_bid(), -qty * 15 // 8)
        elif overall_spread < -boundary:
            pc_mgr.make_sell_order(pc_mgr.highest_bid(), -qty)
            c_mgr.make_buy_order(c_mgr.lowest_ask(), qty * 15 // 8)


class PicnicTrader:
    olivia_signal_ukulele: int

    def __init__(self):
        self.olivia_signal_ukulele = 0

    def run(self, s: State):
        bk_mgr = s.order_manager(PICNIC_BASKET)
        bg_mgr = s.order_manager(BAGUETTE)
        dip_mgr = s.order_manager(DIP)
        uke_mgr = s.order_manager(UKULELE)

        mt, ot = s.state.market_trades, s.state.own_trades
        for trade in mt.get(UKULELE, []) + ot.get(UKULELE, []):
            if trade.buyer == OLIVIA:
                self.olivia_signal_ukulele = 1
            elif trade.seller == OLIVIA:
                self.olivia_signal_ukulele = -1

        # Trade on basket vs. price of components
        c = 0.00125
        boundary = 45
        exponent = 2
        overall_spread = (
                (bk_mgr.mid_price() - 367)
                - 4 * dip_mgr.mid_price()
                - 2 * bg_mgr.mid_price()
                - uke_mgr.mid_price()
        )
        qty = round(c * (abs(overall_spread) ** exponent))
        if overall_spread > boundary:
            bk_mgr.make_sell_order(bk_mgr.highest_bid(), -qty)
        elif overall_spread < -boundary:
            bk_mgr.make_buy_order(bk_mgr.lowest_ask(), qty)

        # Trade on dip vs. historic dip mean
        dip_mean = 7075
        dip_spread = dip_mean - dip_mgr.mid_price()
        dip_boundary = 15
        dip_c = 0.5
        if dip_spread > dip_boundary:
            dip_mgr.make_buy_order(dip_mgr.lowest_ask(), int(dip_spread * dip_c))
        elif dip_spread < -dip_boundary:
            dip_mgr.make_sell_order(dip_mgr.highest_bid(), int(dip_spread * dip_c))

        # Trade on 2 baguette / 1 ukulele ETF
        etf_mean = 45240
        etf_spread = etf_mean - 2 * bg_mgr.mid_price() + uke_mgr.mid_price()
        etf_boundary = 100
        etf_c = 0.3

        if self.olivia_signal_ukulele == 1:
            uke_mgr.make_buy_order(uke_mgr.lowest_ask(), uke_mgr.buy_quota)
        if self.olivia_signal_ukulele == -1:
            uke_mgr.make_sell_order(uke_mgr.highest_bid(), uke_mgr.sell_quota)

        if etf_spread > etf_boundary:
            bg_mgr.make_buy_order(bg_mgr.lowest_ask(), 2 * int(etf_spread * etf_c))
            if self.olivia_signal_ukulele == 0:
                uke_mgr.make_buy_order(uke_mgr.lowest_ask(), int(etf_spread * etf_c))
        elif etf_spread < -etf_boundary:
            bg_mgr.make_sell_order(bg_mgr.highest_bid(), 2 * int(etf_spread * etf_c))
            if self.olivia_signal_ukulele == 0:
                uke_mgr.make_sell_order(uke_mgr.highest_bid(), int(etf_spread * etf_c))


class BananaTrader:
    def __init__(self):
        self.olivia_signal_banana = 0

    def run(self, s: State):
        mgr = s.order_manager(BANANAS)
        if (mp := mgr.mid_price()) is None:
            return

        mt, ot = s.state.market_trades, s.state.own_trades
        for trade in mt.get(BANANAS, []) + ot.get(BANANAS, []):
            if trade.buyer == OLIVIA:
                self.olivia_signal_banana = 1
            elif trade.seller == OLIVIA:
                self.olivia_signal_banana = -1

        # Trade with all bids above fair price
        buy_orders = mgr.buy_orders()
        for bid in sorted(buy_orders.keys(), reverse=True):
            qty = buy_orders[bid]
            if bid > mp:
                mgr.make_sell_order(bid, -qty)

        # Trade with all asks below fair price
        sell_orders = mgr.sell_orders()
        for ask in sorted(sell_orders.keys()):
            qty = sell_orders[ask]
            if ask < mp:
                mgr.make_buy_order(ask, -qty)

        # Submit extra buy/sell orders
        mgr.make_buy_order(round(mp - 1), mgr.buy_quota)
        mgr.make_sell_order(round(mp + 1), mgr.sell_quota)


class GearTrader:
    # Store the past hour (~416 ticks) of sightings. This allows us to 'look
    # into the future' by one hour and get an estimate of the price at
    # any point in that window.
    n: int
    sightings: list[int]
    target_positions: dict[Callable, int]

    def __init__(self):
        self.n = 416  # ~1 hour, in ticks
        self.sightings = []
        self.target_positions = dict()
        self.target_positions[self.run_directional] = 0
        self.target_positions[self.run_spread_reversion] = 0
        self.target_positions[self.run_signal_detection] = 0

    def run_directional(self):
        """
        Assume that the price of diving gear in the next ~400 ticks can be
        estimated due to the lag between the number of dolphins sightings
        and the price of diving gear.
        """
        this = self.run_directional
        if len(self.sightings) < self.n:
            return

        y = [16 * x + 50315 for x in self.sightings[-self.n:]]
        w = Parameter("gear_dir_weight", default=20).get()
        m, *_ = np.polyfit(range(len(y)), y, deg=1)
        if m < -0.4:
            self.target_positions[this] = -w
        elif m < -0.25:
            self.target_positions[this] = -w // 2
        elif m < 0.25:
            self.target_positions[this] = 0
        if m < 0.4:
            self.target_positions[this] = w // 2
        else:
            self.target_positions[this] = w

        logger.print(f"[GEAR] Predicted price slope: {m}")

    def run_spread_reversion(self, s: State):
        """
        Assume that the spread between the diving gear price and the estimate
        derived by dolphin sightings tends to revert to 0.
        """
        this = self.run_spread_reversion
        mgr = s.order_manager(GEAR)
        if (mid := mgr.mid_price()) is None:
            return

        fair_price = 16 * self.sightings[-1] + 50315
        spread = fair_price - mid
        sigma = 205
        c = Parameter("gear_sigma_c", default=1).get()
        w = Parameter("gear_spread_weight", default=40).get()
        if abs(spread) >= c * sigma:
            self.target_positions[this] = np.sign(spread) * w
        else:
            self.target_positions[this] = 0

        logger.print(f"[GEAR] Spread: {spread}")

    def run_signal_detection(self):
        """
        Look for recent sharp movements in dolphin sightings and react as
        fast as possible.
        """
        this = self.run_signal_detection
        recent_sightings = self.sightings[-20:]
        self.target_positions[this] = 0
        for x1, x2 in zip(recent_sightings, recent_sightings[1:]):
            if abs(x2 - x1) > 5:
                self.target_positions[this] = np.sign(x2 - x1) * 50
                logger.print(f"[GEAR] Signal detected: {x2 - x1}")

    def run(self, s: State):
        x = s.state.observations[DOLPHINS]
        self.sightings.append(x)

        self.run_directional()
        self.run_spread_reversion(s)
        self.run_signal_detection()

        # Consolidate target positions across the methods, clamp to [-50, 50]
        target_position = 0
        for method_target in self.target_positions.values():
            target_position += method_target
        target_position = sorted((-50, target_position, 50))[1]

        # Calculate position delta. If all methods agree on the sign of the
        # target position or the position is getting pushed to the boundary,
        # move more aggressively.
        pos = s.state.position.get(GEAR, 0)
        delta = target_position - pos
        quantity = int(np.sign(delta)) * 2
        signs = np.sign(list(self.target_positions.values()))
        if len(set(signs)) == 1 or abs(target_position) == 50:
            quantity *= 5

        # Try to reach target position. If the target position isn't super
        # opinionated, and it's close enough, don't eat the bid/ask spread
        # and leave the position as-is.
        logger.print(f"[GEAR] Position: {pos} | Target: {target_position}")
        if abs(delta) > 10 or abs(target_position) > 10:
            bid = s.order_manager(GEAR).highest_bid()
            ask = s.order_manager(GEAR).lowest_ask()
            if bid is not None and delta > 0:
                s.order_manager(GEAR).make_buy_order(ask, quantity)
            if ask is not None and delta < 0:
                s.order_manager(GEAR).make_sell_order(bid, quantity)


class BerryTrader:
    sell_signal: bool
    buy_signal: bool

    def __init__(self):
        self.sell_signal = False
        self.buy_signal = False

    def run_normal(self, s: State):
        mgr = s.order_manager(BERRIES)
        bid = mgr.highest_bid()
        ask = mgr.lowest_ask()
        mp = mgr.mid_price()
        pos = s.state.position.get(BERRIES, 0)

        # Buy before the peak
        if 400000 <= s.state.timestamp < 500000 and ask is not None:
            mgr.make_buy_order(ask, mgr.buy_quota)

        # Sell at the peak
        elif 500000 <= s.state.timestamp < 700000 and bid is not None:
            mgr.make_sell_order(bid, mgr.sell_quota)

        # Reset position to 0 after the peak
        elif (
                700000 <= s.state.timestamp <= 710000
                and ask is not None
                and s.state.position.get(BERRIES, 0) < 0
        ):
            mgr.make_buy_order(ask, min(mgr.buy_quota, -pos))

        # Market-make everywhere else
        elif mp is not None:
            mid_price = round(mp)
            bp = mid_price - 3 if pos >= 100 else mid_price - 2
            sp = mid_price + 3 if pos <= -100 else mid_price + 2
            mgr.make_buy_order(bp, 10)
            mgr.make_sell_order(sp, -10)

    def run(self, s: State):
        mgr = s.order_manager(BERRIES)
        bid = mgr.highest_bid()
        ask = mgr.lowest_ask()

        mt, ot = s.state.market_trades, s.state.own_trades
        for trade in mt.get(BERRIES, []) + ot.get(BERRIES, []):
            if trade.buyer == OLIVIA:
                self.buy_signal = True
                self.sell_signal = False
            if trade.seller == OLIVIA:
                self.sell_signal = True
                self.buy_signal = False

        if not self.sell_signal and not self.buy_signal:
            self.run_normal(s)
        else:
            if self.sell_signal and bid is not None:
                mgr.make_sell_order(bid, mgr.sell_quota)
            if self.buy_signal and ask is not None:
                mgr.make_buy_order(ask, mgr.buy_quota)


class Trader:
    pearl_trader: PearlTrader
    banana_trader: BananaTrader
    coco_pina_trader: CocoPinaTrader
    berry_trader: BerryTrader
    gear_trader: GearTrader
    picnic_trader: PicnicTrader

    logger: Logger

    def __init__(self):
        self.pearl_trader = PearlTrader()
        self.banana_trader = BananaTrader()
        self.coco_pina_trader = CocoPinaTrader()
        self.berry_trader = BerryTrader()
        self.gear_trader = GearTrader()
        self.picnic_trader = PicnicTrader()
        self.logger = logger

    def run(self, state: TradingState):
        s = State(state)
        self.pearl_trader.run(s)
        self.banana_trader.run(s)
        self.coco_pina_trader.run(s)
        self.gear_trader.run(s)
        self.berry_trader.run(s)
        self.picnic_trader.run(s)

        orders = {
            PEARLS: s.order_manager(PEARLS).pending_orders(),
            BANANAS: s.order_manager(BANANAS).pending_orders(),
            COCOS: s.order_manager(COCOS).pending_orders(),
            PINAS: s.order_manager(PINAS).pending_orders(),
            BERRIES: s.order_manager(BERRIES).pending_orders(),
            GEAR: s.order_manager(GEAR).pending_orders(),
            DIP: s.order_manager(DIP).pending_orders(),
            BAGUETTE: s.order_manager(BAGUETTE).pending_orders(),
            UKULELE: s.order_manager(UKULELE).pending_orders(),
            PICNIC_BASKET: s.order_manager(PICNIC_BASKET).pending_orders(),
        }
        logger.flush(state, orders)
        return orders


logger = Logger(local=True)
