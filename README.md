# 🤖 Kraken Swing Trading Bot - Automated

Bot de trading automatizado que opera en Kraken usando señales de swing structure de Larry Williams. Datos de yfinance, ejecución automática cada hora con GitHub Actions.

## ⚠️ ADVERTENCIA

- **Opera con dinero real en Kraken**
- El margen amplifica ganancias Y pérdidas
- Empieza SIEMPRE con `DRY_RUN=true` (simulación)
- Nunca inviertas más de lo que puedes perder

## 🎯 Características

✅ Datos históricos de yfinance (gratis, sin límites)  
✅ Detección automática de swing points (intermediate level)  
✅ Gestión automática de posiciones con:
  - Stop Loss (4% por defecto)
  - Take Profit (8% por defecto)
  - Trailing Stop (2.5% desde 3% ganancia)
✅ Filtro de volumen  
✅ Notificaciones Telegram  
✅ Ejecución automática cada hora  
✅ Modo simulación para testing

## 📋 Requisitos

1. **Cuenta Kraken** con margen habilitado
2. **API Keys de Kraken** con permisos:
   - Query Funds ✅
   - Query Open Orders & Trades ✅
   - Create & Modify Orders ✅
   - Cancel/Close Orders ✅
3. **Bot de Telegram** (opcional pero recomendado)
4. **Repositorio GitHub** con Actions habilitado

## 🚀 Setup Rápido

### 1. Configurar Kraken

1. Kraken → Settings → API
2. Generate New Key
3. Seleccionar permisos (ver arriba)
4. Guardar API Key y Secret

### 2. Configurar Telegram (opcional)

1. Telegram → [@BotFather](https://t.me/botfather) → `/newbot`
2. Guardar token
3. [@userinfobot](https://t.me/userinfobot) para obtener tu Chat ID

### 3. Configurar GitHub

1. Fork este repositorio
2. Settings → Secrets and variables → Actions
3. Agregar estos secrets:

**OBLIGATORIOS:**
- `KRAKEN_API_KEY` - Tu API key de Kraken
- `KRAKEN_API_SECRET` - Tu API secret de Kraken

**OPCIONALES:**
- `TELEGRAM_BOT_TOKEN` - Token del bot
- `TELEGRAM_CHAT_ID` - Tu chat ID

### 4. Estructura de archivos

```
tu-repo/
├── .github/
│   └── workflows/
│       └── trading-bot.yml
├── kraken_yfinance_bot.py
├── requirements.txt
└── README.md
```

### 5. Primera ejecución

1. Actions → Kraken Trading Bot → Run workflow
2. Dejar `dry_run: true` (simulación)
3. Verificar logs
4. Si todo OK, cambiar a `dry_run: false` en el workflow

## ⚙️ Configuración

Edita variables de entorno en `.github/workflows/trading-bot.yml`:

```yaml
env:
  # Trading
  TRADING_SYMBOL: 'ADA-USD'    # yfinance symbol
  KRAKEN_PAIR: 'ADAEUR'        # Par en Kraken
  POSITION_SIZE_PCT: '0.30'    # 30% del capital
  LEVERAGE: '3'                # 3x leverage
  MIN_BALANCE: '10.0'          # Balance mínimo
  
  # Risk Management
  STOP_LOSS_PCT: '4.0'         # Stop loss 4%
  TAKE_PROFIT_PCT: '8.0'       # Take profit 8%
  TRAILING_STOP_PCT: '2.5'     # Trailing 2.5%
  MIN_PROFIT_FOR_TRAILING: '3.0'  # Activar trailing desde 3%
  
  # Strategy
  LOOKBACK_PERIOD: '90d'       # Historia: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y
  CANDLE_INTERVAL: '1h'        # Velas: 1h, 4h, 1d
  USE_VOLUME_FILTER: 'true'    # Filtro de volumen
  
  # Mode
  DRY_RUN: 'true'              # false = REAL
```

## 📊 Funcionamiento

### Cada hora, el bot:

1. **Descarga** datos de yfinance (últimos 90 días)
2. **Detecta** swing points (intermediate level)
3. **Verifica** posiciones abiertas:
   - Aplica stop loss, take profit, trailing stop
   - Cierra si se activa algún stop
4. **Si no hay posiciones**:
   - Busca nueva señal BUY/SELL
   - Abre posición si hay señal
5. **Notifica** todo a Telegram

### Ejemplo con 40€:

```
Balance: 40€
Position size: 30% = 12€
Leverage: 3x = 36€ efectivos
Precio ADA: 0.30€
Cantidad: 120 ADA

Stop Loss: -4% = cierra en -1.44€ de pérdida
Take Profit: +8% = cierra en +2.88€ de ganancia
Trailing Stop: desde +3%, retroceso 2.5%
```

## 📱 Notificaciones

Recibirás mensajes sobre:
- 🟢 Nuevas posiciones abiertas
- 🔴 Posiciones cerradas (con razón)
- ⚠️ Advertencias
- ❌ Errores

## 🔧 Ajustes Comunes

### Cambiar frecuencia de ejecución

Edita el `cron` en `trading-bot.yml`:

```yaml
schedule:
  - cron: '0 * * * *'       # Cada hora
  # - cron: '0 */2 * * *'   # Cada 2 horas
  # - cron: '0 */4 * * *'   # Cada 4 horas
  # - cron: '0 9,15,21 * * *'  # 9am, 3pm, 9pm
```

### Cambiar crypto

```yaml
TRADING_SYMBOL: 'SOL-USD'   # yfinance
KRAKEN_PAIR: 'SOLEUR'       # Kraken
```

Cryptos populares:
- BTC: `BTC-USD` / `XBTEUR`
- ETH: `ETH-USD` / `ETHEUR`
- SOL: `SOL-USD` / `SOLEUR`
- ADA: `ADA-USD` / `ADAEUR`

### Más/menos riesgo

```yaml
# Conservador
POSITION_SIZE_PCT: '0.20'   # 20%
LEVERAGE: '2'               # 2x
STOP_LOSS_PCT: '3.0'        # Stop 3%

# Agresivo
POSITION_SIZE_PCT: '0.40'   # 40%
LEVERAGE: '5'               # 5x
STOP_LOSS_PCT: '5.0'        # Stop 5%
```

## 🐛 Troubleshooting

**"Insufficient funds"**
- Balance < MIN_BALANCE
- Deposita más o reduce MIN_BALANCE

**"No se pudieron descargar datos"**
- yfinance temporalmente caído
- Esperar próxima ejecución

**Bot no abre posiciones**
- No detecta señales de swing
- Normal, puede tomar días
- Revisar logs para ver swing points

**Telegram no funciona**
- Verifica token y chat_id
- Envía `/start` a tu bot
- El bot seguirá funcionando sin Telegram

## 📈 Monitoreo

1. **GitHub Actions**: Ver historial y logs
2. **Telegram**: Notificaciones en tiempo real
3. **Kraken**: Verificar posiciones y balance

## ⚠️ Consideraciones

- El bot NO hace análisis fundamental
- Opera SOLO con estructura de precio (swing points)
- Puede tener rachas perdedoras
- Revisa backtest primero (ver archivos anteriores)
- Empieza con capital pequeño
- Monitorea regularmente

## 🔐 Seguridad

- API keys solo en GitHub Secrets (nunca en código)
- Permisos mínimos necesarios en Kraken
- Deshabilita withdrawal en API keys
- Usa IP whitelisting en Kraken si es posible

## 📝 Licencia

MIT - Usa bajo tu propio riesgo

---

**¿Dudas?** Revisa los logs en Actions → Selecciona ejecución → Expande steps
