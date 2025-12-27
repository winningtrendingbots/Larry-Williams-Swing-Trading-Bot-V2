"""
KRAKEN SWING BOT - YFINANCE + KRAKEN TRADING
Version completa con gestión automática de posiciones
"""

import os
import time
import hmac
import hashlib
import base64
import urllib.parse
from datetime import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════
#                          CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    # Kraken
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
    KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET', '')
    KRAKEN_API_URL = 'https://api.kraken.com'
    
    # Trading
    TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'ADA-USD')  # yfinance
    KRAKEN_PAIR = os.getenv('KRAKEN_PAIR', 'ADAEUR')  # Kraken
    POSITION_SIZE_PCT = float(os.getenv('POSITION_SIZE_PCT', '0.30'))
    LEVERAGE = int(os.getenv('LEVERAGE', '3'))
    MIN_BALANCE = float(os.getenv('MIN_BALANCE', '10.0'))
    
    # Risk Management
    USE_STOP_LOSS = os.getenv('USE_STOP_LOSS', 'true').lower() == 'true'
    STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '4.0'))
    USE_TAKE_PROFIT = os.getenv('USE_TAKE_PROFIT', 'true').lower() == 'true'
    TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '8.0'))
    USE_TRAILING_STOP = os.getenv('USE_TRAILING_STOP', 'true').lower() == 'true'
    TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '2.5'))
    MIN_PROFIT_FOR_TRAILING = float(os.getenv('MIN_PROFIT_FOR_TRAILING', '3.0'))
    
    # Strategy
    LOOKBACK_PERIOD = os.getenv('LOOKBACK_PERIOD', '90d')  # yfinance: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y
    CANDLE_INTERVAL = os.getenv('CANDLE_INTERVAL', '1h')  # 1h, 4h, 1d
    USE_VOLUME_FILTER = os.getenv('USE_VOLUME_FILTER', 'true').lower() == 'true'
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Mode
    DRY_RUN = os.getenv('DRY_RUN', 'true').lower() == 'true'


# ═══════════════════════════════════════════════════════════════════════════
#                        KRAKEN CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class KrakenClient:
    def __init__(self, api_key: str, api_secret: str, api_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_url = api_url
        self.session = requests.Session()
    
    def _sign(self, urlpath: str, data: dict) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()
    
    def _request(self, endpoint: str, data: dict = None, private: bool = False) -> dict:
        url = self.api_url + endpoint
        
        if private:
            data = data or {}
            data['nonce'] = int(time.time() * 1000)
            headers = {
                'API-Key': self.api_key,
                'API-Sign': self._sign(endpoint, data)
            }
            response = self.session.post(url, data=data, headers=headers, timeout=30)
        else:
            response = self.session.get(url, params=data, timeout=30)
        
        response.raise_for_status()
        result = response.json()
        
        if result.get('error') and len(result['error']) > 0:
            raise Exception(f"Kraken error: {result['error']}")
        
        return result.get('result', {})
    
    def get_balance(self) -> Tuple[float, str]:
        """Retorna (balance, currency)."""
        result = self._request('/0/private/Balance', private=True)
        balances = {k: float(v) for k, v in result.items()}
        
        fiat = {'ZUSD': 'USD', 'USD': 'USD', 'ZEUR': 'EUR', 'EUR': 'EUR'}
        
        for key, currency in fiat.items():
            if key in balances and balances[key] > 0:
                return balances[key], currency
        
        return 0.0, 'EUR'
    
    def get_open_positions(self) -> dict:
        """Retorna posiciones abiertas."""
        try:
            result = self._request('/0/private/OpenPositions', private=True)
            return result
        except Exception as e:
            if "No open positions" in str(e):
                return {}
            raise
    
    def place_order(self, pair: str, order_type: str, volume: float, leverage: int = None) -> dict:
        """Coloca orden de mercado."""
        data = {
            'pair': pair,
            'type': order_type,
            'ordertype': 'market',
            'volume': str(volume)
        }
        
        if leverage and leverage > 1:
            data['leverage'] = str(leverage)
        
        return self._request('/0/private/AddOrder', data=data, private=True)
    
    def close_position(self, position_id: str) -> dict:
        """Cierra posición por ID."""
        data = {'txid': position_id, 'type': 'market'}
        return self._request('/0/private/ClosePosition', data=data, private=True)


# ═══════════════════════════════════════════════════════════════════════════
#                        TELEGRAM
# ═══════════════════════════════════════════════════════════════════════════

class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{token}"
    
    def send(self, message: str) -> bool:
        if not self.token or not self.chat_id:
            print(f"📱 {message}")
            return False
        
        try:
            if len(message) > 4000:
                message = message[:3900] + "\n..."
            
            data = {'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}
            response = requests.post(f"{self.api_url}/sendMessage", data=data, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Telegram error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
#                        SWING DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

def calculate_volume_ma(data: pd.DataFrame, period: int = 20) -> pd.Series:
    return data['Volume'].rolling(window=period).mean()

class SwingDetector:
    def __init__(self, data: pd.DataFrame, volume_filter: bool = True):
        self.data = data.copy()
        self.volume_filter = volume_filter
        self.volume_ma = calculate_volume_ma(data) if volume_filter else None
        
        self.st_highs = pd.Series(index=data.index, dtype=float)
        self.st_lows = pd.Series(index=data.index, dtype=float)
        self.int_highs = pd.Series(index=data.index, dtype=float)
        self.int_lows = pd.Series(index=data.index, dtype=float)
    
    def _check_volume(self, i: int) -> bool:
        if not self.volume_filter or self.volume_ma is None:
            return True
        
        if pd.isna(self.volume_ma.iloc[i]):
            return True
        
        return self.data['Volume'].iloc[i] > self.volume_ma.iloc[i]
    
    def detect(self):
        """Detecta todos los swing points."""
        highs = self.data['High'].values
        lows = self.data['Low'].values
        
        # Short-term
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                if self._check_volume(i):
                    self.st_lows.iloc[i] = lows[i]
        
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                if self._check_volume(i):
                    self.st_highs.iloc[i] = highs[i]
        
        # Intermediate
        st_high_idx = self.st_highs.dropna().index.tolist()
        for i in range(1, len(st_high_idx) - 1):
            p, c, n = st_high_idx[i-1], st_high_idx[i], st_high_idx[i+1]
            
            if self.st_highs[c] > self.st_highs[p] and self.st_highs[c] > self.st_highs[n]:
                self.int_highs[c] = self.st_highs[c]
        
        st_low_idx = self.st_lows.dropna().index.tolist()
        for i in range(1, len(st_low_idx) - 1):
            p, c, n = st_low_idx[i-1], st_low_idx[i], st_low_idx[i+1]
            
            if self.st_lows[c] < self.st_lows[p] and self.st_lows[c] < self.st_lows[n]:
                self.int_lows[c] = self.st_lows[c]
    
    def get_signal(self) -> Tuple[Optional[str], Optional[float]]:
        """Retorna última señal (BUY/SELL, precio)."""
        self.detect()
        
        highs = self.int_highs.dropna()
        lows = self.int_lows.dropna()
        
        if len(highs) == 0 and len(lows) == 0:
            return None, None
        
        last_high = highs.index[-1] if len(highs) > 0 else pd.Timestamp.min
        last_low = lows.index[-1] if len(lows) > 0 else pd.Timestamp.min
        
        if last_low > last_high:
            return 'BUY', lows.iloc[-1]
        elif last_high > last_low:
            return 'SELL', highs.iloc[-1]
        else:
            return None, None


# ═══════════════════════════════════════════════════════════════════════════
#                        POSITION MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class PositionManager:
    def __init__(self, config: Config, kraken: KrakenClient, telegram: Telegram):
        self.config = config
        self.kraken = kraken
        self.telegram = telegram
        self.peak_prices = {}  # position_id -> peak_price
    
    def check_position(self, pos_id: str, pos_data: dict, current_price: float) -> Tuple[bool, str]:
        """
        Analiza posición y decide si cerrar.
        Returns: (should_close, reason)
        """
        pos_type = pos_data.get('type', 'long')
        entry_price = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        leverage = float(pos_data.get('leverage', 1))
        
        # PnL
        if pos_type == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
        
        print(f"   {pos_type.upper()}: entrada ${entry_price:.4f}, actual ${current_price:.4f}")
        print(f"   PnL: {pnl_pct:+.2f}%")
        
        # Stop Loss
        if self.config.USE_STOP_LOSS and pnl_pct <= -self.config.STOP_LOSS_PCT:
            return True, f"🛑 Stop Loss: {pnl_pct:.2f}%"
        
        # Take Profit
        if self.config.USE_TAKE_PROFIT and pnl_pct >= self.config.TAKE_PROFIT_PCT:
            return True, f"🎯 Take Profit: {pnl_pct:.2f}%"
        
        # Trailing Stop
        if self.config.USE_TRAILING_STOP and pnl_pct >= self.config.MIN_PROFIT_FOR_TRAILING:
            # Actualizar peak
            if pos_id not in self.peak_prices:
                self.peak_prices[pos_id] = current_price
            
            if pos_type == 'long' and current_price > self.peak_prices[pos_id]:
                self.peak_prices[pos_id] = current_price
            elif pos_type == 'short' and current_price < self.peak_prices[pos_id]:
                self.peak_prices[pos_id] = current_price
            
            # Calcular retroceso desde peak
            peak = self.peak_prices[pos_id]
            if pos_type == 'long':
                peak_pnl = ((peak - entry_price) / entry_price) * 100 * leverage
            else:
                peak_pnl = ((entry_price - peak) / entry_price) * 100 * leverage
            
            drawdown = peak_pnl - pnl_pct
            
            if drawdown >= self.config.TRAILING_STOP_PCT:
                return True, f"📉 Trailing Stop: peak {peak_pnl:.2f}%, actual {pnl_pct:.2f}%"
        
        return False, ""
    
    def close_position(self, pos_id: str, reason: str, pos_data: dict, current_price: float):
        """Cierra posición."""
        print(f"\n🔴 Cerrando posición: {pos_id}")
        print(f"   Razón: {reason}")
        
        if not self.config.DRY_RUN:
            try:
                result = self.kraken.close_position(pos_id)
                print(f"   ✓ Cerrada: {result}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                self.telegram.send(f"❌ Error cerrando posición: {e}")
                return
        else:
            print(f"   🧪 [SIMULACIÓN]")
        
        # Limpiar tracking
        if pos_id in self.peak_prices:
            del self.peak_prices[pos_id]
        
        # Notificar
        pos_type = pos_data.get('type', 'unknown')
        entry = float(pos_data.get('cost', 0)) / float(pos_data.get('vol', 1))
        
        msg = f"""
🔴 <b>POSICIÓN CERRADA</b>

<b>Tipo:</b> {pos_type.upper()}
<b>Entrada:</b> ${entry:.4f}
<b>Salida:</b> ${current_price:.4f}

<b>Razón:</b> {reason}
<b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        if self.config.DRY_RUN:
            msg = "🧪 <b>SIMULACIÓN</b>\n" + msg
        
        self.telegram.send(msg)


# ═══════════════════════════════════════════════════════════════════════════
#                        TRADING BOT
# ═══════════════════════════════════════════════════════════════════════════

class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.kraken = KrakenClient(config.KRAKEN_API_KEY, config.KRAKEN_API_SECRET, config.KRAKEN_API_URL)
        self.telegram = Telegram(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        self.position_mgr = PositionManager(config, self.kraken, self.telegram)
    
    def get_market_data(self) -> pd.DataFrame:
        """Descarga datos de yfinance."""
        print(f"\n📊 Descargando {self.config.TRADING_SYMBOL} ({self.config.LOOKBACK_PERIOD}, {self.config.CANDLE_INTERVAL})...")
        
        ticker = yf.Ticker(self.config.TRADING_SYMBOL)
        data = ticker.history(period=self.config.LOOKBACK_PERIOD, interval=self.config.CANDLE_INTERVAL)
        
        if data.empty:
            raise Exception("No se pudieron descargar datos")
        
        print(f"✓ {len(data)} velas descargadas")
        return data
    
    def open_position(self, signal: str, current_price: float, reason: str):
        """Abre nueva posición."""
        try:
            balance, currency = self.kraken.get_balance()
            
            if balance < self.config.MIN_BALANCE:
                print(f"⚠️  Balance insuficiente: {balance:.2f} {currency}")
                return
            
            capital = balance * self.config.POSITION_SIZE_PCT
            effective = capital * self.config.LEVERAGE
            volume = effective / current_price
            volume = round(volume, 2)
            
            if volume <= 0:
                print("⚠️  Volumen = 0")
                return
            
            print(f"\n🟢 Abriendo {signal}")
            print(f"   Capital: {capital:.2f} {currency} (x{self.config.LEVERAGE})")
            print(f"   Volumen: {volume} @ ${current_price:.4f}")
            
            if not self.config.DRY_RUN:
                order_type = 'buy' if signal == 'BUY' else 'sell'
                result = self.kraken.place_order(
                    pair=self.config.KRAKEN_PAIR,
                    order_type=order_type,
                    volume=volume,
                    leverage=self.config.LEVERAGE
                )
                print(f"   ✓ Ejecutada: {result}")
            else:
                print(f"   🧪 [SIMULACIÓN]")
            
            # Notificar
            msg = f"""
🟢 <b>NUEVA POSICIÓN</b>

<b>Par:</b> {self.config.KRAKEN_PAIR}
<b>Tipo:</b> {signal}
<b>Precio:</b> ${current_price:.4f}
<b>Cantidad:</b> {volume}
<b>Leverage:</b> {self.config.LEVERAGE}x

<b>Razón:</b> {reason}
<b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            if self.config.DRY_RUN:
                msg = "🧪 <b>SIMULACIÓN</b>\n" + msg
            
            self.telegram.send(msg)
            
        except Exception as e:
            print(f"❌ Error abriendo: {e}")
            self.telegram.send(f"❌ Error: {e}")
    
    def run(self):
        """Ciclo principal."""
        print("\n" + "="*70)
        print("KRAKEN SWING BOT - YFINANCE + AUTO TRADING")
        print("="*70)
        print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Símbolo: {self.config.TRADING_SYMBOL} → {self.config.KRAKEN_PAIR}")
        print(f"Modo: {'🧪 SIMULACIÓN' if self.config.DRY_RUN else '💰 REAL'}")
        print("="*70)
        
        try:
            # 1. Verificar posiciones abiertas
            print("\n📊 Verificando posiciones...")
            positions = self.kraken.get_open_positions()
            
            # Obtener precio actual
            data = self.get_market_data()
            current_price = float(data['Close'].iloc[-1])
            print(f"💰 Precio actual: ${current_price:.4f}")
            
            if positions:
                print(f"✓ {len(positions)} posición(es) abierta(s)")
                
                for pos_id, pos_data in positions.items():
                    should_close, reason = self.position_mgr.check_position(pos_id, pos_data, current_price)
                    
                    if should_close:
                        self.position_mgr.close_position(pos_id, reason, pos_data, current_price)
                    else:
                        print(f"   ✓ Mantener posición")
                
                print("\nℹ️  Posiciones abiertas, no buscar nuevas señales")
                return
            
            print("✓ No hay posiciones abiertas")
            
            # 2. Buscar señal
            print("\n🔍 Detectando swing points...")
            detector = SwingDetector(data, volume_filter=self.config.USE_VOLUME_FILTER)
            signal, signal_price = detector.get_signal()
            
            if signal is None:
                print("ℹ️  No hay señales")
                return
            
            print(f"✓ Señal: {signal} @ ${signal_price:.4f}")
            
            # 3. Abrir posición
            reason = f"Señal {signal} detectada (intermediate swings)"
            self.open_position(signal, current_price, reason)
            
            print("\n✅ Ciclo completado")
            
        except Exception as e:
            msg = f"Error: {str(e)}"
            print(f"\n❌ {msg}")
            self.telegram.send(f"❌ {msg}")
            raise


# ═══════════════════════════════════════════════════════════════════════════
#                              MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    config = Config()
    
    if not config.KRAKEN_API_KEY or not config.KRAKEN_API_SECRET:
        print("❌ Faltan credenciales Kraken")
        return
    
    bot = TradingBot(config)
    bot.run()


if __name__ == "__main__":
    main()
