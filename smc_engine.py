"""
SMC/ICT Crypto Signal Engine
Detects: Market Structure, FVG, Order Blocks, Liquidity Sweeps, Displacement
Only fires when confidence >= 90%
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class Trend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


class Phase(Enum):
    ACCUMULATION = "Accumulation"
    DISTRIBUTION = "Distribution"
    EXPANSION = "Expansion"
    REVERSAL = "Reversal"


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str  # HH, HL, LH, LL


@dataclass
class FVG:
    top: float
    bottom: float
    direction: str  # bullish / bearish
    index: int
    filled: bool = False

    @property
    def midpoint(self):
        return (self.top + self.bottom) / 2


@dataclass
class OrderBlock:
    top: float
    bottom: float
    direction: str  # bullish / bearish
    index: int
    tested: bool = False
    valid: bool = True

    @property
    def midpoint(self):
        return (self.top + self.bottom) / 2


@dataclass
class LiquiditySweep:
    price: float
    direction: str  # buy_side / sell_side
    index: int
    swept: bool = True


@dataclass
class Signal:
    symbol: str
    timeframe: str
    direction: str        # LONG / SHORT
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    rr: float
    confidence: float
    phase: str
    trend_htf: str
    trapped: str
    smart_money_zone: str
    fvg: Optional[FVG]
    ob: Optional[OrderBlock]
    sweep: Optional[LiquiditySweep]
    institution_verdict: str
    institution_price: Optional[float]
    pro_agreement: int        # out of 100
    bulk_verdict: str         # BUY NOW / HOLD / SELL NOW
    reasons: List[str] = field(default_factory=list)


class SMCEngine:
    """
    Institutional Smart Money Concepts Analysis Engine
    """

    def __init__(self, lookback: int = 100, atr_multiplier: float = 1.5):
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier

    # ─────────────────────────────────────────
    # 1. MARKET STRUCTURE
    # ─────────────────────────────────────────
    def _find_swings(self, df: pd.DataFrame, window: int = 5) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Identify swing highs and lows using fractal logic"""
        highs, lows = [], []
        for i in range(window, len(df) - window):
            hi = df['high'].iloc[i]
            lo = df['low'].iloc[i]
            if all(hi >= df['high'].iloc[i - j] for j in range(1, window + 1)) and \
               all(hi >= df['high'].iloc[i + j] for j in range(1, window + 1)):
                highs.append(SwingPoint(index=i, price=hi, kind="SH"))
            if all(lo <= df['low'].iloc[i - j] for j in range(1, window + 1)) and \
               all(lo <= df['low'].iloc[i + j] for j in range(1, window + 1)):
                lows.append(SwingPoint(index=i, price=lo, kind="SL"))
        return highs, lows

    def _classify_structure(self, highs: List[SwingPoint], lows: List[SwingPoint]) -> Tuple[Trend, List[SwingPoint]]:
        """Classify market structure and label HH/HL/LH/LL"""
        labeled = []
        if len(highs) < 2 or len(lows) < 2:
            return Trend.SIDEWAYS, labeled

        # Last 3 swing highs and lows
        rh = sorted(highs[-3:], key=lambda x: x.index)
        rl = sorted(lows[-3:], key=lambda x: x.index)

        hh_count = sum(1 for i in range(1, len(rh)) if rh[i].price > rh[i-1].price)
        lh_count = sum(1 for i in range(1, len(rh)) if rh[i].price < rh[i-1].price)
        hl_count = sum(1 for i in range(1, len(rl)) if rl[i].price > rl[i-1].price)
        ll_count = sum(1 for i in range(1, len(rl)) if rl[i].price < rl[i-1].price)

        bull_score = hh_count + hl_count
        bear_score = lh_count + ll_count

        if bull_score > bear_score and bull_score >= 2:
            trend = Trend.BULLISH
        elif bear_score > bull_score and bear_score >= 2:
            trend = Trend.BEARISH
        else:
            trend = Trend.SIDEWAYS

        # Label last swing points
        for i in range(1, len(rh)):
            kind = "HH" if rh[i].price > rh[i-1].price else "LH"
            labeled.append(SwingPoint(rh[i].index, rh[i].price, kind))
        for i in range(1, len(rl)):
            kind = "HL" if rl[i].price > rl[i-1].price else "LL"
            labeled.append(SwingPoint(rl[i].index, rl[i].price, kind))

        return trend, labeled

    # ─────────────────────────────────────────
    # 2. FAIR VALUE GAPS
    # ─────────────────────────────────────────
    def _find_fvgs(self, df: pd.DataFrame, min_size_pct: float = 0.001) -> List[FVG]:
        fvgs = []
        for i in range(2, len(df)):
            c1_low = df['low'].iloc[i - 2]
            c1_high = df['high'].iloc[i - 2]
            c3_low = df['low'].iloc[i]
            c3_high = df['high'].iloc[i]

            # Bullish FVG: gap between candle 1 high and candle 3 low
            if c3_low > c1_high:
                size = (c3_low - c1_high) / c1_high
                if size >= min_size_pct:
                    fvgs.append(FVG(top=c3_low, bottom=c1_high, direction="bullish", index=i))

            # Bearish FVG: gap between candle 3 high and candle 1 low
            if c3_high < c1_low:
                size = (c1_low - c3_high) / c3_high
                if size >= min_size_pct:
                    fvgs.append(FVG(top=c1_low, bottom=c3_high, direction="bearish", index=i))

        # Mark filled FVGs
        current_price = df['close'].iloc[-1]
        for fvg in fvgs:
            if fvg.direction == "bullish" and current_price < fvg.bottom:
                fvg.filled = True
            elif fvg.direction == "bearish" and current_price > fvg.top:
                fvg.filled = True

        return [f for f in fvgs if not f.filled]

    # ─────────────────────────────────────────
    # 3. ORDER BLOCKS
    # ─────────────────────────────────────────
    def _find_order_blocks(self, df: pd.DataFrame, trend: Trend) -> List[OrderBlock]:
        obs = []
        for i in range(1, len(df) - 1):
            c = df.iloc[i]
            next_c = df.iloc[i + 1]

            # Bullish OB: last bearish candle before strong bullish move
            if c['close'] < c['open'] and next_c['close'] > next_c['open']:
                displacement = (next_c['close'] - next_c['open']) / next_c['open']
                if displacement > 0.002:  # meaningful displacement
                    obs.append(OrderBlock(
                        top=c['open'],
                        bottom=c['low'],
                        direction="bullish",
                        index=i
                    ))

            # Bearish OB: last bullish candle before strong bearish move
            if c['close'] > c['open'] and next_c['close'] < next_c['open']:
                displacement = (next_c['open'] - next_c['close']) / next_c['open']
                if displacement > 0.002:
                    obs.append(OrderBlock(
                        top=c['high'],
                        bottom=c['close'],
                        direction="bearish",
                        index=i
                    ))

        # Keep only last 5 OBs per direction
        bull_obs = [o for o in obs if o.direction == "bullish"][-5:]
        bear_obs = [o for o in obs if o.direction == "bearish"][-5:]
        return bull_obs + bear_obs

    # ─────────────────────────────────────────
    # 4. LIQUIDITY SWEEPS
    # ─────────────────────────────────────────
    def _find_liquidity_sweeps(self, df: pd.DataFrame, highs: List[SwingPoint], lows: List[SwingPoint]) -> List[LiquiditySweep]:
        sweeps = []
        if len(df) < 3:
            return sweeps

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Check if last candle swept a swing high (sell-side liquidity above)
        for sh in highs[-5:]:
            if last_candle['high'] > sh.price and last_candle['close'] < sh.price:
                sweeps.append(LiquiditySweep(price=sh.price, direction="buy_side", index=len(df)-1))

        # Check if last candle swept a swing low (buy-side liquidity below)
        for sl in lows[-5:]:
            if last_candle['low'] < sl.price and last_candle['close'] > sl.price:
                sweeps.append(LiquiditySweep(price=sl.price, direction="sell_side", index=len(df)-1))

        return sweeps

    # ─────────────────────────────────────────
    # 5. DISPLACEMENT CHECK
    # ─────────────────────────────────────────
    def _check_displacement(self, df: pd.DataFrame, direction: str, window: int = 3) -> bool:
        """Check if there's a strong impulsive move (displacement) recently"""
        if len(df) < window + 1:
            return False

        recent = df.iloc[-(window):]
        avg_range = df['high'].iloc[-20:].values - df['low'].iloc[-20:].values
        avg_candle = np.mean(avg_range)

        for i in range(len(recent)):
            c = recent.iloc[i]
            candle_body = abs(c['close'] - c['open'])
            if direction == "bullish" and c['close'] > c['open'] and candle_body > avg_candle * 1.5:
                return True
            if direction == "bearish" and c['close'] < c['open'] and candle_body > avg_candle * 1.5:
                return True
        return False

    # ─────────────────────────────────────────
    # 6. ATR FOR SL SIZING
    # ─────────────────────────────────────────
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    # ─────────────────────────────────────────
    # 7. PREMIUM / DISCOUNT
    # ─────────────────────────────────────────
    def _premium_discount(self, df: pd.DataFrame) -> str:
        recent_high = df['high'].iloc[-50:].max()
        recent_low = df['low'].iloc[-50:].min()
        mid = (recent_high + recent_low) / 2
        cmp = df['close'].iloc[-1]
        equilibrium_band = (recent_high - recent_low) * 0.1

        if cmp > mid + equilibrium_band:
            return "PREMIUM"
        elif cmp < mid - equilibrium_band:
            return "DISCOUNT"
        else:
            return "EQUILIBRIUM"

    # ─────────────────────────────────────────
    # 8. MARKET PHASE
    # ─────────────────────────────────────────
    def _detect_phase(self, df: pd.DataFrame, trend: Trend, sweeps: List[LiquiditySweep]) -> Phase:
        if sweeps:
            return Phase.REVERSAL
        volatility = df['close'].pct_change().std()
        if volatility < 0.005:
            return Phase.ACCUMULATION if trend == Trend.BULLISH else Phase.DISTRIBUTION
        return Phase.EXPANSION

    # ─────────────────────────────────────────
    # 9. CONFIDENCE SCORING
    # ─────────────────────────────────────────
    def _score_confidence(
        self,
        trend: Trend,
        trend_htf: Trend,
        sweep: Optional[LiquiditySweep],
        fvg: Optional[FVG],
        ob: Optional[OrderBlock],
        displacement: bool,
        zone: str,
        direction: str,
        mss: bool
    ) -> Tuple[float, List[str]]:
        score = 0.0
        reasons = []

        # Base — trend alignment
        if trend == trend_htf and trend != Trend.SIDEWAYS:
            score += 20
            reasons.append(f"✅ HTF + LTF trend aligned ({trend.value})")
        elif trend != Trend.SIDEWAYS:
            score += 8
            reasons.append(f"⚠️ Trend present but HTF not fully aligned")

        # Liquidity sweep (critical)
        if sweep:
            score += 25
            reasons.append(f"✅ Liquidity sweep at {sweep.price:.4f} ({sweep.direction})")
        else:
            reasons.append("❌ No liquidity sweep — high-probability entry blocked")

        # Market structure shift
        if mss:
            score += 20
            reasons.append("✅ Market Structure Shift confirmed")
        else:
            reasons.append("❌ No MSS — waiting for break of structure")

        # FVG in zone
        if fvg:
            score += 15
            reasons.append(f"✅ FVG present ({fvg.direction}) [{fvg.bottom:.4f} – {fvg.top:.4f}]")

        # Order Block
        if ob:
            score += 10
            reasons.append(f"✅ Order Block ({ob.direction}) [{ob.bottom:.4f} – {ob.top:.4f}]")

        # Displacement
        if displacement:
            score += 10
            reasons.append("✅ Displacement (impulsive move) detected")
        else:
            reasons.append("❌ No displacement yet")

        # Premium/Discount alignment
        if (direction == "LONG" and zone == "DISCOUNT") or (direction == "SHORT" and zone == "PREMIUM"):
            score += 10
            reasons.append(f"✅ Price in {zone} zone — institutional entry logic valid")
        elif zone == "EQUILIBRIUM":
            score -= 15
            reasons.append("❌ Price in Equilibrium — NO TRADE (middle of range)")

        return min(score, 100.0), reasons

    # ─────────────────────────────────────────
    # 10. INSTITUTION VERDICT
    # ─────────────────────────────────────────
    def _institution_verdict(
        self,
        confidence: float,
        ob: Optional[OrderBlock],
        sweep: Optional[LiquiditySweep],
        displacement: bool,
        mss: bool
    ) -> Tuple[str, Optional[float]]:
        if ob and sweep and displacement and mss:
            return "They are taking a new position just now", ob.midpoint
        elif ob and not displacement:
            return "They are waiting for more confirmation", ob.midpoint
        elif confidence < 70:
            return "No position yet", None
        else:
            return "Institution already in existing position", ob.midpoint if ob else None

    # ─────────────────────────────────────────
    # MAIN ANALYZE FUNCTION
    # ─────────────────────────────────────────
    def analyze(
        self,
        df_ltf: pd.DataFrame,
        df_htf: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Optional[Signal]:
        """
        Full SMC analysis. Returns Signal if confidence >= 90, else None.
        df must have columns: open, high, low, close, volume (lowercase)
        """
        if len(df_ltf) < 50 or len(df_htf) < 50:
            return None

        # ─── HTF Trend ───
        highs_htf, lows_htf = self._find_swings(df_htf, window=5)
        trend_htf, _ = self._classify_structure(highs_htf, lows_htf)

        # ─── LTF Structure ───
        highs_ltf, lows_ltf = self._find_swings(df_ltf, window=3)
        trend_ltf, labeled = self._classify_structure(highs_ltf, lows_ltf)

        # Reject sideways
        if trend_htf == Trend.SIDEWAYS or trend_ltf == Trend.SIDEWAYS:
            return None

        # ─── Direction ───
        direction = "LONG" if trend_htf == Trend.BULLISH else "SHORT"

        # ─── Zone Check ───
        zone = self._premium_discount(df_ltf)
        if zone == "EQUILIBRIUM":
            return None

        # Reject if direction doesn't match zone
        if direction == "LONG" and zone == "PREMIUM":
            return None
        if direction == "SHORT" and zone == "DISCOUNT":
            return None

        # ─── Liquidity Sweeps ───
        sweeps = self._find_liquidity_sweeps(df_ltf, highs_ltf, lows_ltf)
        relevant_sweep = None
        for s in sweeps:
            if direction == "LONG" and s.direction == "sell_side":
                relevant_sweep = s
            elif direction == "SHORT" and s.direction == "buy_side":
                relevant_sweep = s

        # ─── FVGs ───
        fvgs = self._find_fvgs(df_ltf)
        cmp = df_ltf['close'].iloc[-1]
        relevant_fvg = None
        for fvg in sorted(fvgs, key=lambda x: abs(x.midpoint - cmp)):
            if direction == "LONG" and fvg.direction == "bullish" and fvg.top < cmp:
                relevant_fvg = fvg
                break
            elif direction == "SHORT" and fvg.direction == "bearish" and fvg.bottom > cmp:
                relevant_fvg = fvg
                break

        # ─── Order Blocks ───
        obs = self._find_order_blocks(df_ltf, trend_ltf)
        relevant_ob = None
        for ob in sorted(obs, key=lambda x: abs(x.midpoint - cmp)):
            if direction == "LONG" and ob.direction == "bullish" and ob.top < cmp:
                relevant_ob = ob
                break
            elif direction == "SHORT" and ob.direction == "bearish" and ob.bottom > cmp:
                relevant_ob = ob
                break

        # ─── Displacement ───
        disp_direction = "bullish" if direction == "LONG" else "bearish"
        displacement = self._check_displacement(df_ltf, disp_direction)

        # ─── MSS (Market Structure Shift) ───
        mss = False
        if labeled:
            last_labeled = sorted(labeled, key=lambda x: x.index)[-1]
            if direction == "LONG" and last_labeled.kind == "HL":
                mss = True
            elif direction == "SHORT" and last_labeled.kind == "LH":
                mss = True

        # ─── Confidence Score ───
        confidence, reasons = self._score_confidence(
            trend_ltf, trend_htf, relevant_sweep,
            relevant_fvg, relevant_ob, displacement, zone, direction, mss
        )

        # ─── REJECT if < 90 ───
        if confidence < 90:
            return None

        # ─── ATR for sizing ───
        atr = self._calculate_atr(df_ltf)

        # ─── Entry / SL / TP ───
        if direction == "LONG":
            entry = relevant_ob.midpoint if relevant_ob else (relevant_fvg.midpoint if relevant_fvg else cmp)
            sl = entry - (atr * self.atr_multiplier)
            if relevant_ob:
                sl = min(sl, relevant_ob.bottom - atr * 0.3)
            risk = entry - sl
            tp1 = entry + risk * 2
            tp2 = entry + risk * 3
            tp3 = entry + risk * 5
        else:
            entry = relevant_ob.midpoint if relevant_ob else (relevant_fvg.midpoint if relevant_fvg else cmp)
            sl = entry + (atr * self.atr_multiplier)
            if relevant_ob:
                sl = max(sl, relevant_ob.top + atr * 0.3)
            risk = sl - entry
            tp1 = entry - risk * 2
            tp2 = entry - risk * 3
            tp3 = entry - risk * 5

        rr = (tp2 - entry) / (entry - sl) if direction == "LONG" else (entry - tp2) / (sl - entry)
        rr = abs(rr)

        if rr < 2.0:
            return None

        # ─── Phase ───
        phase = self._detect_phase(df_ltf, trend_ltf, sweeps)

        # ─── Trapped traders ───
        trapped = "Retail longs trapped above swing high" if direction == "SHORT" else "Retail shorts trapped below swing low"

        # ─── Institution Verdict ───
        inst_verdict, inst_price = self._institution_verdict(
            confidence, relevant_ob, relevant_sweep, displacement, mss
        )

        # ─── Pro Agreement ───
        pro_agreement = min(int(confidence), 97)

        # ─── Bulk Verdict ───
        if direction == "LONG" and cmp <= entry * 1.002:
            bulk = "🟢 BUY NOW"
        elif direction == "SHORT" and cmp >= entry * 0.998:
            bulk = "🔴 SELL NOW"
        else:
            bulk = "🟡 HOLD — Wait for price to return to zone"

        smart_money_zone = f"OB + FVG confluence at {entry:.4f}" if (relevant_ob and relevant_fvg) else \
                           (f"OB at {relevant_ob.midpoint:.4f}" if relevant_ob else
                            (f"FVG at {relevant_fvg.midpoint:.4f}" if relevant_fvg else "Near structure level"))

        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry=round(entry, 6),
            sl=round(sl, 6),
            tp1=round(tp1, 6),
            tp2=round(tp2, 6),
            tp3=round(tp3, 6),
            rr=round(rr, 2),
            confidence=confidence,
            phase=phase.value,
            trend_htf=trend_htf.value,
            trapped=trapped,
            smart_money_zone=smart_money_zone,
            fvg=relevant_fvg,
            ob=relevant_ob,
            sweep=relevant_sweep,
            institution_verdict=inst_verdict,
            institution_price=inst_price,
            pro_agreement=pro_agreement,
            bulk_verdict=bulk,
            reasons=reasons
        )
