import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class StoreMeger(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        stores_df = pd.read_csv("data/stores.csv")

        # stores_df_transformed = stores_df.drop(["city", "state", "type"], axis=1)
        merged_store_df = pd.merge(X, stores_df, on=["store_nbr", "store_nbr"])
        return merged_store_df


class HolidayMerger(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        holidays_df = pd.read_csv("data/holidays_events.csv")
        holidays_df["date"] = pd.to_datetime(holidays_df["date"])
        holidays_filtered = holidays_df[~holidays_df["transferred"]].copy()
        holidays_filtered["is_holiday"] = True
        holidays_filtered = holidays_filtered.drop(
            ["type", "description", "transferred"], axis=1
        )
        local_holidays = holidays_filtered[holidays_filtered.locale == "Local"].copy()
        regional_holidays = holidays_filtered[
            holidays_filtered.locale == "Regional"
        ].copy()
        national_holidays = holidays_filtered[
            holidays_filtered.locale == "National"
        ].copy()
        national_indexes = pd.merge(X, national_holidays, on="date", how="left")[
            "is_holiday"
        ]
        regional_indexes = pd.merge(
            X,
            regional_holidays,
            left_on=["date", "state"],
            right_on=["date", "locale_name"],
            how="left",
        )["is_holiday"]
        local_indexes = pd.merge(
            X,
            local_holidays,
            left_on=["date", "city"],
            right_on=["date", "locale_name"],
            how="left",
        )["is_holiday"]
        final_indexes = national_indexes | regional_indexes | local_indexes
        X["is_holiday"] = final_indexes
        return X


class OilMerger(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fill_nan_with_mean_from_prev_and_next(self, col):
        df_filled_forward = col.ffill()
        df_filled_backward = col.bfill()
        return (df_filled_forward + df_filled_backward) / 2

    def transform(self, X):
        oil_df = pd.read_csv("data/oil.csv")
        oil_df.columns = ["date", "oil_price"]
        oil_df["date"] = pd.to_datetime(oil_df["date"])
        complete_dates = pd.date_range(start="2013-01-01", end="2017-08-31")
        complete_df = pd.DataFrame({"date": complete_dates})
        merged_days_oil_df = pd.merge(complete_df, oil_df, on="date", how="left")
        merged_days_oil_df["oil_price"] = self.fill_nan_with_mean_from_prev_and_next(
            merged_days_oil_df["oil_price"]
        )
        merged_days_oil_df = merged_days_oil_df.ffill().bfill()

        return pd.merge(X, merged_days_oil_df, on=["date", "date"], how="left")


class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["date"] = pd.to_datetime(X["date"])
        X["month"] = X["date"].dt.month
        X["day"] = X["date"].dt.day
        X["year"] = X["date"].dt.year
        X["day_of_week"] = X["date"].dt.dayofweek
        X["is_weekend"] = X["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(["id"], axis=1)
        # X = X.drop(["date"], axis=1)
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        family = X["family"]
        X = pd.get_dummies(
            X, columns=["family", "city", "state", "type"], drop_first=True
        )
        X["family"] = family
        return X


def cap_sales(s, cap=10000):
    return s.map(lambda x: x if x < cap else cap)


def add_lagged_features(group, lags):
    for lag in lags:
        group[f"lagged_sales_{lag}"] = group["sales"].shift(lag)
    return group.reset_index(drop=True)


def add_rolling_mean(group, window_sizes):
    for window_size in window_sizes:
        group[f"rolling_mean_{window_size}_days"] = (
            group["sales"].rolling(window_size).mean()
        )
    return group


def update_time_features(data, full_data=None):
    # Lagged Features
    grouped_data = data.groupby(["store_nbr", "family"]).apply(
        add_lagged_features, lags=[1, 2]
    )
    grouped_data.reset_index(drop=True, inplace=True)
    updated_data = pd.merge(
        data,
        grouped_data[
            ["store_nbr", "family", "date", "lagged_sales_1", "lagged_sales_2"]
        ],
        on=["store_nbr", "family", "date"],
    )

    # Rolling Window Statistics
    grouped_data = updated_data.groupby(["store_nbr", "family"]).apply(
        add_rolling_mean, window_sizes=[14, 28]
    )
    grouped_data.reset_index(drop=True, inplace=True)
    updated_data = pd.merge(
        updated_data,
        grouped_data[
            [
                "store_nbr",
                "family",
                "date",
                "rolling_mean_14_days",
                "rolling_mean_28_days",
            ]
        ],
        on=["store_nbr", "family", "date"],
    )
    if full_data is not None:
        updated_data = pd.merge(
            full_data,
            updated_data,
            on=["store_nbr", "family", "date"],
        )
    return updated_data


data_transform_pipeline = Pipeline(
    [
        ("transform_dates", DateTransformer()),
        ("merge_stores", StoreMeger()),
        ("merge_holidays", HolidayMerger()),
        ("merge_oil_prices", OilMerger()),
        ("one_hot_encoding", OneHotEncoder()),
        ("column_dropper", ColumnDropper()),
        # ("min_max_scale", MinMaxScaler()),
    ]
)
