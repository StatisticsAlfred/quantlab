# QuantLab v3 - Momentum + Teknisk Analys + Hybrid Builder
# Kräver:
# pip install streamlit yfinance pandas numpy plotly

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from nordic_universe import NORDIC_STOCKS

st.set_page_config(
    page_title="QuantLab Strategy Builder",
    layout="wide"
)

st.title("📈 QuantLab Strategy Builder")

strategy_type = st.selectbox(
    "Strategi",
    [
        "Momentum",
        "Teknisk Analys",
        "Hybrid"
    ]
)

st.sidebar.header("Momentum")

mom1_weight = st.sidebar.slider(
    "1M Momentum %",
    0,
    100,
    0
)

mom3_weight = st.sidebar.slider(
    "3M Momentum %",
    0,
    100,
    20
)

mom6_weight = st.sidebar.slider(
    "6M Momentum %",
    0,
    100,
    30
)

mom12_weight = st.sidebar.slider(
    "12M Momentum %",
    0,
    100,
    50
)

st.sidebar.header("Tekniska Filter")

use_ma50 = st.sidebar.checkbox(
    "Pris > MA50",
    value=True
)

use_ma200 = st.sidebar.checkbox(
    "Pris > MA200",
    value=True
)

use_rsi = st.sidebar.checkbox(
    "RSI > 50",
    value=True
)

use_macd = st.sidebar.checkbox(
    "MACD > 0",
    value=True
)

use_low_volatility = st.sidebar.checkbox(
    "Premiera låg volatilitet",
    value=True
)

st.sidebar.header("Hybridmodell")

momentum_weight = st.sidebar.slider(
    "Momentum Vikt %",
    0,
    100,
    60
)

technical_weight = st.sidebar.slider(
    "Teknisk Vikt %",
    0,
    100,
    40
)

st.sidebar.header("Portfölj")

top_n = st.sidebar.slider(
    "Antal innehav",
    5,
    20,
    5
)

if st.button("Kör Screening"):

    results = []

    progress = st.progress(0)

    for i, stock in enumerate(NORDIC_STOCKS):

        try:

            data = yf.download(
                stock,
                period="3y",
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                continue

            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            if len(close) < 252:
                continue

            latest = close.iloc[-1]

            # ========================
            # MOMENTUM
            # ========================

            mom1 = (
                close.iloc[-1] /
                close.iloc[-21]
                - 1
            ) * 100

            mom3 = (
                close.iloc[-1] /
                close.iloc[-63]
                - 1
            ) * 100

            mom6 = (
                close.iloc[-1] /
                close.iloc[-126]
                - 1
            ) * 100

            mom12 = (
                close.iloc[-1] /
                close.iloc[-252]
                - 1
            ) * 100

            momentum_score = (

                mom1 * (mom1_weight / 100)

                +

                mom3 * (mom3_weight / 100)

                +

                mom6 * (mom6_weight / 100)

                +

                mom12 * (mom12_weight / 100)

            )

            # ========================
            # VOLATILITET
            # ========================

            returns = close.pct_change()

            annual_volatility = (
                returns.std()
                * np.sqrt(252)
                * 100
            )

            risk_adjusted_momentum = (
                momentum_score
                /
                max(
                    annual_volatility,
                    1
                )
            )

            # ========================
            # MA FILTER
            # ========================

            technical_score = 0

            ma50 = close.rolling(
                50
            ).mean()

            ma200 = close.rolling(
                200
            ).mean()

            if (
                use_ma50
                and latest >
                ma50.iloc[-1]
            ):
                technical_score += 25

            if (
                use_ma200
                and latest >
                ma200.iloc[-1]
            ):
                technical_score += 25

            # ========================
            # RSI
            # ========================

            delta = close.diff()

            gain = delta.clip(
                lower=0
            )

            loss = (
                -delta.clip(
                    upper=0
                )
            )

            avg_gain = (
                gain
                .rolling(14)
                .mean()
            )

            avg_loss = (
                loss
                .rolling(14)
                .mean()
            )

            rs = avg_gain / avg_loss

            rsi = 100 - (
                100 /
                (1 + rs)
            )

            latest_rsi = (
                rsi.iloc[-1]
            )

            if (
                use_rsi
                and latest_rsi > 50
            ):
                technical_score += 25

            # ========================
            # MACD
            # ========================

            ema12 = close.ewm(
                span=12,
                adjust=False
            ).mean()

            ema26 = close.ewm(
                span=26,
                adjust=False
            ).mean()

            macd = (
                ema12 - ema26
            )

            latest_macd = (
                macd.iloc[-1]
            )

            if (
                use_macd
                and latest_macd > 0
            ):
                technical_score += 25

            # ========================
            # HYBRID SCORE
            # ========================

            if strategy_type == "Momentum":

                total_score = (
                    risk_adjusted_momentum
                )

            elif strategy_type == "Teknisk Analys":

                total_score = (
                    technical_score
                )

            else:

                total_score = (

                    risk_adjusted_momentum
                    *
                    (
                        momentum_weight
                        / 100
                    )

                    +

                    technical_score
                    *
                    (
                        technical_weight
                        / 100
                    )

                )

            if use_low_volatility:

                total_score = (
                    total_score
                    -
                    annual_volatility
                    * 0.10
                )

            results.append(
                {
                    "Aktie": stock,

                    "1M %": round(
                        mom1,
                        2
                    ),

                    "3M %": round(
                        mom3,
                        2
                    ),

                    "6M %": round(
                        mom6,
                        2
                    ),

                    "12M %": round(
                        mom12,
                        2
                    ),

                    "RSI": round(
                        float(
                            latest_rsi
                        ),
                        1
                    ),

                    "MACD": round(
                        float(
                            latest_macd
                        ),
                        2
                    ),

                    "Volatilitet %": round(
                        annual_volatility,
                        2
                    ),

                    "Teknisk Score": round(
                        technical_score,
                        2
                    ),

                    "Momentum Score": round(
                        risk_adjusted_momentum,
                        2
                    ),

                    "Total Score": round(
                        total_score,
                        2
                    )
                }
            )

        except Exception:
            pass

        progress.progress(
            (i + 1)
            /
            len(NORDIC_STOCKS)
        )

    ranking = pd.DataFrame(
        results
    )

    if ranking.empty:

        st.error(
            "Ingen data kunde hämtas."
        )

    else:

        ranking = ranking.sort_values(
            "Total Score",
            ascending=False
        )

        st.subheader(
            "🏆 Full Ranking"
        )

        st.dataframe(
            ranking,
            use_container_width=True
        )

        st.subheader(
            f"⭐ Top {top_n} Kandidater"
        )

        st.table(
            ranking.head(top_n)
        )

        st.metric(
            "Bästa Aktie",
            ranking.iloc[0]["Aktie"]
        )

        st.metric(
            "Högsta Score",
            round(
                float(
                    ranking.iloc[0]["Total Score"]
                ),
                2
            )
        )

        st.subheader(
            "Portföljförslag"
        )

        portfolio = (
            ranking
            .head(top_n)
            [["Aktie", "Total Score"]]
            .copy()
        )

        portfolio["Vikt"] = round(
            100 / top_n,
            2
        )

        st.dataframe(
            portfolio,
            use_container_width=True
        )