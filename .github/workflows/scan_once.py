"""
scan_once.py — GitHub Actions এর জন্য
প্রতিবার চললে একটা full scan করে Telegram-এ signal পাঠায়
"""

import asyncio
import os
import ccxt
import pandas as pd
from datetime import datetime
from telegram import Bot
from smc_engine import SMCEngine, Signal

# ─── CONFIG FROM ENVIRONMENT ─────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
EXCHANGE_ID        = os.environ.get("EXCHANGE_ID", "binance")

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "MATIC/USDT",
]

TIMEFRAMES = ["15m", "1h", "4h"]

HTF_MAP = {
    "15m": "4h",
    "1h":  "1d",
    "4h":  "1w",
}

MIN_CONFIDENCE = 90


# ─── FETCH DATA ───────────────────────────────────────────────
def fetch_ohlcv(exchange, symbol, timeframe, limit=200):
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ Fetch error {symbol} {timeframe}: {e}")
        return None


# ─── SIGNAL FORMATTER ────────────────────────────────────────
def format_signal(signal: Signal) -> str:
    direction_emoji = "🟢📈 LONG" if signal.direction == "LONG" else "🔴📉 SHORT"
    bar = "█" * int(signal.confidence / 10) + "░" * (10 - int(signal.confidence / 10))
    inst_price = f"`{signal.institution_price:.4f}`" if signal.institution_price else "Unknown"

    return f"""
╔══════════════════════════════╗
║   🏛️  INSTITUTIONAL SIGNAL   ║
╚══════════════════════════════╝

📊 *{signal.symbol}* | `{signal.timeframe}`
{direction_emoji}

━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 *TRADE PLAN*
━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Entry:     `{signal.entry}`
🛑 Stop Loss: `{signal.sl}`
🏆 TP1 (2R):  `{signal.tp1}`
🏆 TP2 (3R):  `{signal.tp2}`
🏆 TP3 (5R):  `{signal.tp3}`
📐 R:R Ratio: `1 : {signal.rr}`

━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 *SMC ANALYSIS*
━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Phase:     {signal.phase}
📌 HTF Trend: {signal.trend_htf}
📌 Zone:      {signal.smart_money_zone}
🪤 Trapped:   {signal.trapped}

━━━━━━━━━━━━━━━━━━━━━━━━━━
🏛️ *INSTITUTION VERDICT*
━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 {signal.institution_verdict}
💵 Their Price: {inst_price}
{signal.bulk_verdict}

━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 *CONFIDENCE*
━━━━━━━━━━━━━━━━━━━━━━━━━━
Score: `{signal.confidence:.1f}%` [{bar}]
👥 Pro Agreement: `{signal.pro_agreement}/100`

━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ *CONFLUENCE*
━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(signal.reasons)}

━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Risk max 1% capital
🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
""".strip()


# ─── MAIN SCAN ────────────────────────────────────────────────
async def main():
    engine   = SMCEngine()
    exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    bot      = Bot(token=TELEGRAM_BOT_TOKEN)

    print(f"🔍 Scanning {len(SYMBOLS)} symbols × {len(TIMEFRAMES)} timeframes...")
    signals_found = []

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            htf = HTF_MAP.get(tf, "1d")

            df_ltf = fetch_ohlcv(exchange, symbol, tf)
            df_htf = fetch_ohlcv(exchange, symbol, htf)

            if df_ltf is None or df_htf is None:
                continue

            signal = engine.analyze(df_ltf, df_htf, symbol, tf)

            if signal and signal.confidence >= MIN_CONFIDENCE:
                print(f"🎯 SIGNAL FOUND: {symbol} {tf} {signal.direction} {signal.confidence:.1f}%")
                signals_found.append(signal)

            await asyncio.sleep(0.5)

    # ─── SEND SIGNALS ────────────────────────────────────────
    if signals_found:
        for sig in signals_found:
            msg = format_signal(sig)
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode="Markdown"
            )
            print(f"✅ Sent: {sig.symbol} {sig.timeframe}")
            await asyncio.sleep(1)
    else:
        scanned = len(SYMBOLS) * len(TIMEFRAMES)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=(
                f"🔍 *Scan Complete* — {datetime.utcnow().strftime('%H:%M UTC')}\n"
                f"Scanned {scanned} pairs × {len(TIMEFRAMES)} timeframes\n\n"
                f"No institutional setup ≥ {MIN_CONFIDENCE}% found.\n"
                f"Capital protected. Staying flat. 🛡️"
            ),
            parse_mode="Markdown"
        )
        print("📭 No signals. Notification sent.")


if __name__ == "__main__":
    asyncio.run(main())
