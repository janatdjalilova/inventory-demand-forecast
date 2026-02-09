import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ============================
# Константы
# ============================

BEST_THRESHOLD_POPULAR = 0.59  # оптимальный порог по F1
DATA_PATH = "data/kaspi_coffee_cleaned.csv"
MODEL_POP_PATH = "models/model_popularity_rf.pkl"
MODEL_REG_PATH = "models/model_reviews_rf.pkl"


# ============================
# Helpers: безопасный вывод таблиц (fix ArrowTypeError)
# ============================

def df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Make df Arrow-compatible for Streamlit dataframe rendering."""
    def _to_safe(x):
        if isinstance(x, (bytes, bytearray)):
            try:
                return x.decode("utf-8", errors="ignore")
            except Exception:
                return str(x)
        return x

    out = df.copy()
    out = out.map(_to_safe)
    # Convert everything to string to avoid mixed dtypes issues in Arrow
    return out.astype(str)


# ============================
# Загрузка данных и моделей
# ============================

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "product_id" in df.columns:
        df["product_id"] = df["product_id"].astype(str)
    return df


@st.cache_resource
def load_models(pop_path: str, reg_path: str):
    clf = joblib.load(pop_path)
    reg = joblib.load(reg_path)
    return clf, reg


# ============================
# Inventory-блок
# ============================

def compute_inventory_metrics(
    df: pd.DataFrame,
    reg_model,
    planning_period_days: float = 30.0,
    review_to_units: float = 1.0,
    lead_time_days: float = 7.0,
    service_level_z: float = 1.65,
    demand_cv: float = 0.3,
) -> pd.DataFrame:

    df_inv = df.copy()

    feature_cols_reg = list(reg_model.feature_names_in_)

    # Fail fast if required columns are missing
    missing = [c for c in feature_cols_reg if c not in df_inv.columns]
    if missing:
        raise ValueError(f"Missing columns for regression model: {missing}")

    X_full = df_inv[feature_cols_reg]

    # Прогноз спроса
    df_inv["pred_reviews"] = reg_model.predict(X_full)
    df_inv["pred_reviews"] = df_inv["pred_reviews"].clip(lower=0)

    # Ежедневный спрос
    df_inv["daily_demand_units"] = (
        df_inv["pred_reviews"] / planning_period_days * review_to_units
    )

    # Стандартное отклонение
    df_inv["demand_std_per_day"] = demand_cv * df_inv["daily_demand_units"]

    # Safety stock
    df_inv["safety_stock"] = (
        service_level_z * np.sqrt(lead_time_days) * df_inv["demand_std_per_day"]
    )

    # Reorder point
    df_inv["reorder_point"] = (
        df_inv["daily_demand_units"] * lead_time_days + df_inv["safety_stock"]
    )

    # Годовой спрос
    df_inv["annual_demand"] = df_inv["daily_demand_units"] * 365.0

    # Средний запас
    df_inv["avg_inventory"] = (
        df_inv["reorder_point"] + df_inv["safety_stock"]
    ) / 2.0

    # Inventory turnover (proxy)
    df_inv["inventory_turnover"] = np.where(
        df_inv["reorder_point"] > 0,
        df_inv["annual_demand"] / df_inv["reorder_point"],
        np.nan,
    )

    return df_inv


# ============================
# UI helpers
# ============================

def show_feature_importance(clf, feature_cols):
    fi = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_fi = fi.head(10)

    st.subheader("Feature importance (RandomForest, tuned)")
    fig, ax = plt.subplots(figsize=(6, 4))
    top_fi.sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)


def add_download_button(df_filtered: pd.DataFrame):
    csv = df_filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Скачать таблицу (CSV)",
        data=csv,
        file_name="kaspi_coffee_inventory_filtered.csv",
        mime="text/csv",
    )


# ============================
# Основное приложение
# ============================

def main():
    st.set_page_config(
        page_title="Kaspi Coffee Inventory",
        layout="wide",
    )

    st.sidebar.title("Kaspi Coffee Inventory")
    st.sidebar.write("Demo Day — Жанат Джалилова")
    st.sidebar.write(f"Используемый порог популярности: **{BEST_THRESHOLD_POPULAR:.2f}**")
    st.sidebar.caption("Порог подобран по максимуму F1-score.")

    # Бизнес-параметры
    st.sidebar.header("Business assumptions")
    lead_time_days = st.sidebar.slider("Lead time (days)", 3, 21, 7)
    service_level_z = st.sidebar.slider("Service level Z (~95% = 1.65)", 1.0, 2.5, 1.65, 0.05)
    demand_cv = st.sidebar.slider("Demand CV", 0.1, 0.8, 0.3, 0.05)
    st.sidebar.caption("Используется в расчёте safety stock и reorder point.")

    mode = st.sidebar.radio(
        "Режим",
        ["Single SKU analysis", "Portfolio dashboard"],
    )

    # Данные и модели
    df = load_data(DATA_PATH)
    clf, reg = load_models(MODEL_POP_PATH, MODEL_REG_PATH)
    feature_cols = list(clf.feature_names_in_)

    # Inventory-метрики для всех SKU
    df_inventory = compute_inventory_metrics(
        df,
        reg_model=reg,
        planning_period_days=30.0,
        review_to_units=1.0,
        lead_time_days=float(lead_time_days),
        service_level_z=float(service_level_z),
        demand_cv=float(demand_cv),
    )

    # Fail fast if required columns are missing
    missing_clf = [c for c in feature_cols if c not in df_inventory.columns]
    if missing_clf:
        raise ValueError(f"Missing columns for classification model: {missing_clf}")

    # Прогноз популярности
    X_all = df_inventory[feature_cols]
    df_inventory["proba_popular"] = clf.predict_proba(X_all)[:, 1]
    df_inventory["is_popular_ml"] = (
        df_inventory["proba_popular"] >= BEST_THRESHOLD_POPULAR
    ).astype(int)

    # Лейбл для селектора
    df_inventory["sku_label"] = (
        df_inventory["name"].astype(str).str.slice(0, 60)
        + " | "
        + df_inventory["product_id"].astype(str)
    )

    # ====================================
    # MODE 1: SINGLE SKU
    # ====================================
    if mode == "Single SKU analysis":
        st.title("Single SKU Explorer")

        selected_label = st.selectbox(
            "Выберите товар (SKU):",
            options=df_inventory["sku_label"].sort_values().tolist(),
        )

        row = df_inventory[df_inventory["sku_label"] == selected_label].iloc[0]

        # Подготовка данных
        X_row = row[feature_cols].to_frame().T
        proba_pop = clf.predict_proba(X_row)[0, 1]
        pred_reviews = float(reg.predict(X_row)[0])
        pred_reviews = max(0.0, pred_reviews)

        # Полный пересчёт inventory для этого SKU
        df_one = pd.DataFrame([row])
        df_one_inv = compute_inventory_metrics(
            df_one,
            reg_model=reg,
            planning_period_days=30,
            review_to_units=1,
            lead_time_days=lead_time_days,
            service_level_z=service_level_z,
            demand_cv=demand_cv,
        )
        inv = df_one_inv.iloc[0]

        st.subheader("ML-прогнозы")
        c1, c2, c3 = st.columns(3)
        c1.metric("Popularity probability", f"{proba_pop:.2f}")
        c2.metric("Predicted reviews (30 days)", f"{pred_reviews:.1f}")
        c3.metric("Popular?", "YES ✅" if proba_pop >= BEST_THRESHOLD_POPULAR else "NO ❌")

        c4, c5, c6 = st.columns(3)
        c4.metric("Daily demand", f"{inv['daily_demand_units']:.2f}")
        c5.metric("Safety stock", f"{inv['safety_stock']:.1f}")
        c6.metric("Reorder point", f"{inv['reorder_point']:.1f}")

        st.markdown("---")
        st.subheader("SKU details")

        # Колонки для подробной таблицы (без дублей)
        detail_cols = [
            "product_id",
            "name",
            "brand_top",
            "price",
            "rating",
        ] + feature_cols + [
            "pred_reviews",
            "daily_demand_units",
            "safety_stock",
            "reorder_point",
            "annual_demand",
            "inventory_turnover",
        ]

        # убираем дубликаты, сохраняя порядок
        seen = set()
        detail_cols_unique = []
        for c in detail_cols:
            if c not in seen and c in df_one_inv.columns:
                seen.add(c)
                detail_cols_unique.append(c)

        # показываем как транспонированную таблицу "поле / значение" (Arrow-safe)
        df_show = df_one_inv[detail_cols_unique].T.rename(columns={df_one_inv.index[0]: "value"})
        st.dataframe(df_for_display(df_show), width="stretch")

    # ====================================
    # MODE 2: PORTFOLIO DASHBOARD
    # ====================================
    else:
        st.title("Portfolio dashboard")

        st.sidebar.header("Фильтры")

        brands = sorted(df_inventory["brand_top"].dropna().unique())
        selected_brands = st.sidebar.multiselect(
            "Бренд",
            options=brands,
            default=brands,
        )

        price_min, price_max = float(df_inventory["price"].min()), float(df_inventory["price"].max())
        price_range = st.sidebar.slider(
            "Price range",
            price_min,
            price_max,
            (price_min, price_max),
            step=100.0,
        )

        rating_min, rating_max = float(df_inventory["rating"].min()), float(df_inventory["rating"].max())
        rating_range = st.sidebar.slider(
            "Rating range",
            rating_min,
            rating_max,
            (rating_min, rating_max),
            step=0.1,
        )

        only_popular = st.sidebar.checkbox("Только популярные (ML)", value=False)

        mask = (
            df_inventory["brand_top"].isin(selected_brands)
            & df_inventory["price"].between(price_range[0], price_range[1])
            & df_inventory["rating"].between(rating_range[0], rating_range[1])
        )
        if only_popular:
            mask &= df_inventory["is_popular_ml"] == 1

        df_f = df_inventory[mask].copy()

        st.subheader("Portfolio KPIs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SKU count", len(df_f))
        c2.metric("Avg rating", f"{df_f['rating'].mean():.2f}")
        c3.metric("Share popular", f"{df_f['is_popular_ml'].mean():.0%}")
        c4.metric("Total annual demand", f"{df_f['annual_demand'].sum():.0f}")

        st.markdown("---")

        st.subheader("Top-10 by predicted demand")
        top_demand = (
            df_f.sort_values("pred_reviews", ascending=False)
            .head(10)[
                [
                    "product_id",
                    "name",
                    "brand_top",
                    "price",
                    "rating",
                    "pred_reviews",
                    "daily_demand_units",
                    "safety_stock",
                    "reorder_point",
                ]
            ]
        )
        st.dataframe(df_for_display(top_demand), width="stretch")

        st.subheader("Bottom-10 by predicted demand")
        bottom_demand = (
            df_f.sort_values("pred_reviews", ascending=True)
            .head(10)[
                [
                    "product_id",
                    "name",
                    "brand_top",
                    "price",
                    "rating",
                    "pred_reviews",
                    "daily_demand_units",
                    "safety_stock",
                    "reorder_point",
                ]
            ]
        )
        st.dataframe(df_for_display(bottom_demand), width="stretch")

        st.subheader("Top-10 by safety stock (stockout risk)")
        top_safety = (
            df_f.sort_values("safety_stock", ascending=False)
            .head(10)[
                [
                    "product_id",
                    "name",
                    "brand_top",
                    "price",
                    "rating",
                    "pred_reviews",
                    "daily_demand_units",
                    "safety_stock",
                    "reorder_point",
                ]
            ]
        )
        st.dataframe(df_for_display(top_safety), width="stretch")

        st.markdown("---")

        show_feature_importance(clf, feature_cols)

        st.subheader("Price vs predicted demand")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df_f["price"], df_f["pred_reviews"], alpha=0.6)
        ax.set_xlabel("Price")
        ax.set_ylabel("Predicted demand (reviews)")
        st.pyplot(fig)

        add_download_button(
            df_f[
                [
                    "product_id",
                    "name",
                    "brand_top",
                    "price",
                    "rating",
                    "proba_popular",
                    "is_popular_ml",
                    "pred_reviews",
                    "daily_demand_units",
                    "safety_stock",
                    "reorder_point",
                    "annual_demand",
                    "inventory_turnover",
                ]
            ]
        )


if __name__ == "__main__":
    main()
