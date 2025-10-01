# Theta-Bot Roadmap

## Vision
Theta-Bot aims to be a **containerized AI-driven trading system** capable of:
- Predicting market movements with advanced models,
- Executing trades with hedging or arbitrage to reduce risk,
- Running flexibly on local hardware (Podman, M1/M4) and in the cloud.

---

## Architecture Overview
- **Core Trading Engine**
  - Interfaces with Binance Spot API
  - Supports multiple symbols (≥2 cryptos + 1 base currency)
  - Trade execution, logging, PnL tracking

- **Prediction Module**
  - Machine learning & deep learning models
  - Methods: Kalman filter, PCA, sentiment analysis
  - Fokker–Planck based probability modeling
  - Jacobi theta function decomposition

- **Risk & Strategy Layer**
  - Classic predictive trading (buy/sell)
  - Paired trades (hedged/entangled orders)
  - Options-aware trading (where available)
  - Arbitrage strategies (intra-exchange, cross-exchange – future scope)

- **Containerization**
  - Podman/Docker deployment
  - Configurable via environment variables
  - Cloud-ready for horizontal scaling

---

## Strategies

### 1. Predictive Trading
- Uses AI models to predict short-term price movements
- Frequent rotation of trades to maximize profit
- Risk: occasional mispredictions (mitigated by hedging)

### 2. Hedging / Arbitrage (No Options)
- Always pairs a BUY with a SELL in correlated pairs
- Target: overall profit even if one leg loses
- Works within spot market, but limited by liquidity & spread

### 3. Options-based Hedging
- Uses options to cap downside risk
- Example: Buy crypto spot, buy protective put option
- Profit = spot gain – option premium
- Needs special care for margin calls

---

## Roadmap

### Phase 1 – Local MVP (1–2 months)
- Containerized trading bot
- Supports 2 cryptos + 1 base currency
- Basic predictive model (moving averages, simple ML)
- Logging & config in YAML/JSON

### Phase 2 – Enhanced Prediction (3–6 months)
- Add Kalman filter, PCA, sentiment analysis
- Integrate Fokker–Planck equation solver
- Train/test deep learning model on Binance data
- Paper trading mode for simulation

### Phase 3 – Hedging & Arbitrage (6–9 months)
- Implement entangled buy/sell orders
- Add hedging strategies without options
- Explore cross-pair arbitrage within Binance

### Phase 4 – Options Integration (9–12 months)
- Implement options-aware trading
- Risk management for margin calls
- Optimize profit vs option cost

### Phase 5 – Cloud Deployment (12+ months)
- Migrate to cloud (Kubernetes/OpenShift or serverless workers)
- Scale across multiple pairs & exchanges
- Combine strategies (AI + hedging + arbitrage)
- Advanced monitoring & alerting

---

## Next Steps
- Finalize container baseline (`theta-bot` running in Podman)
- Implement simple predictive strategy
- Add Roadmap tracking inside repo
