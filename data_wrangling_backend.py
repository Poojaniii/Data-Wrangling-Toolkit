import pandas as pd
import numpy as np

from google.colab import files

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder
)

from scipy.stats import chi2_contingency
from scipy.stats import pointbiserialr
from scipy.stats import f_oneway

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


class PlottingMethods:
    """
    A modular plotting utilities class for generating
    granular chart types (Bar, Pie, Histogram) that return
    HTML-wrapped Plotly figures for flexible embedding.
    """

    @staticmethod
    def bar_chart(df, column):
        """
        Generates a bar chart for the given column.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str): The column name to visualize.

        Returns:
            str or None: HTML string of the bar chart,
                or None if inputs are invalid.
        """

        if df is None or column not in df.columns:
            return None

        counts = df[column].value_counts()

        fig = px.bar(
            x=counts.index,
            y=counts.values,
            title=f"Bar Chart - {column}"
        )

        return fig.to_html()

    @staticmethod
    def pie_chart(df, column):
        """
        Generates a pie chart for the given column.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str): The column name to visualize.

        Returns:
            str or None: HTML string of the pie chart,
                or None if inputs are invalid.
        """

        if df is None or column not in df.columns:
            return None

        counts = df[column].value_counts()

        fig = px.pie(
            names=counts.index,
            values=counts.values,
            title=f"Pie Chart - {column}"
        )

        return fig.to_html()

    @staticmethod
    def histogram(df, column):
        """
        Generates a histogram for the given column.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str): The column name to visualize.

        Returns:
            str or None: HTML string of the histogram,
                or None if inputs are invalid.
        """

        if df is None or column not in df.columns:
            return None

        fig = px.histogram(df, x=column)

        return fig.to_html()


class DataInspector:
    """
    A reusable data inspection class that automates
    CSV data ingestion, advanced cleaning, feature
    engineering preparation, and high-level statistical
    visualization using Plotly.
    """

    def __init__(self):
        """Initializes the DataInspector with no loaded data."""
        self.df = None

    def upload_data(self):
        """
        Uploads a CSV file via Google Colab file upload widget
        and loads it into a pandas DataFrame.

        Automatically handles garbage strings ('?', 'n/a',
        'NULL', ' ') by converting them to NaN, and applies
        auto-type correction to force-convert columns to
        numeric types where possible.

        Returns:
            pd.DataFrame: The loaded and sanitized DataFrame.
        """

        uploaded = files.upload()

        filename = list(uploaded.keys())[0]

        self.df = pd.read_csv(
            filename,
            na_values=[
                '?',
                'n/a',
                'NULL',
                ' '
            ]
        )

        self.auto_type_correction()

        return self.df

    def auto_type_correction(self):
        """
        Attempts to convert each column to a numeric type.

        A column is converted only if the conversion does
        not result in an entirely null column, preserving
        any partial numeric data.
        """

        for col in self.df.columns:

            converted = pd.to_numeric(
                self.df[col],
                errors='coerce'
            )

            if converted.notna().sum() > 0:
                self.df[col] = converted

    def data_summary(self):
        """
        Displays a structural summary of the loaded dataset,
        including row/column counts, a preview of the first
        20 rows, and a breakdown of numerical vs. categorical
        columns.
        """

        print("Rows:", self.df.shape[0])
        print("Columns:", self.df.shape[1])

        print("\nFirst 20 Rows\n")
        display(self.df.head(20))

        num_cols = self.df.select_dtypes(
            include=np.number
        ).columns

        cat_cols = self.df.select_dtypes(
            exclude=np.number
        ).columns

        print("\nNumeric Columns")
        print(list(num_cols))

        print("\nCategorical Columns")
        print(list(cat_cols))

    def handle_missing_values(
            self,
            strategy='mean',
            constant_value=0
    ):
        """
        Fills missing values in the DataFrame using the
        specified imputation strategy.

        Parameters:
            strategy (str): The imputation method to use.
                Options: 'mean', 'median', 'mode', 'constant'.
                Defaults to 'mean'.
            constant_value: The value to use when strategy
                is 'constant'. Defaults to 0.
        """

        for col in self.df.columns:

            if self.df[col].isna().sum() == 0:
                continue

            if strategy == 'mean':

                if pd.api.types.is_numeric_dtype(
                        self.df[col]):
                    self.df[col] = self.df[col].fillna(
                        self.df[col].mean()
                    )

            elif strategy == 'median':

                if pd.api.types.is_numeric_dtype(
                        self.df[col]):
                    self.df[col] = self.df[col].fillna(
                        self.df[col].median()
                    )

            elif strategy == 'mode':

                self.df[col] = self.df[col].fillna(
                    self.df[col].mode()[0]
                )

            elif strategy == 'constant':

                self.df[col] = self.df[col].fillna(
                    constant_value
                )

    def remove_duplicates(self):
        """
        Removes exact duplicate rows from the DataFrame
        and prints the number of rows removed.
        """

        before = len(self.df)

        self.df.drop_duplicates(
            inplace=True
        )

        after = len(self.df)

        print(
            f"Removed {before-after} duplicate rows"
        )

    def handle_outliers(
            self,
            column,
            delete=False
    ):
        """
        Detects outliers in a numeric column using the
        IQR method (1.5 * IQR rule).

        Parameters:
            column (str): The column name to check for
                outliers.
            delete (bool): If True, removes outlier rows
                from the DataFrame. If False, only reports
                the count. Defaults to False.
        """

        q1 = self.df[column].quantile(0.25)

        q3 = self.df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = self.df[
            (self.df[column] < lower)
            |
            (self.df[column] > upper)
        ]

        print(
            "Outliers Found:",
            len(outliers)
        )

        if delete:

            self.df = self.df[
                (self.df[column] >= lower)
                &
                (self.df[column] <= upper)
            ]

    def delete_rows(self):
        """
        Interactively deletes rows from the DataFrame by
        accepting comma-separated row indices from user input.
        """

        rows = input(
            "Enter row indices:"
        )

        rows = [
            int(x)
            for x in rows.split(',')
        ]

        self.df.drop(
            rows,
            inplace=True
        )

    def delete_columns(self):
        """
        Interactively deletes columns from the DataFrame by
        accepting comma-separated column names from user input.
        """

        cols = input(
            "Enter column names:"
        )

        cols = cols.split(',')

        self.df.drop(
            columns=cols,
            inplace=True
        )

    def extract_normalized_numeric_data(
            self,
            method='standard'
    ):
        """
        Extracts and normalizes all numeric columns using
        the specified scaling method.

        Parameters:
            method (str): The scaling method to apply.
                Options: 'minmax', 'standard' (Z-score),
                'robust' (IQR-based). Defaults to 'standard'.

        Returns:
            pd.DataFrame: A new DataFrame with scaled
                numeric columns.
        """

        num_df = self.df.select_dtypes(
            include=np.number
        )

        if method == 'minmax':
            scaler = MinMaxScaler()

        elif method == 'robust':
            scaler = RobustScaler()

        else:
            scaler = StandardScaler()

        scaled = scaler.fit_transform(
            num_df
        )

        return pd.DataFrame(
            scaled,
            columns=num_df.columns
        )

    def extract_normalized_categorical_data(
            self,
            method='onehot'
    ):
        """
        Extracts and encodes all categorical columns using
        the specified encoding method.

        Parameters:
            method (str): The encoding method to apply.
                Options: 'onehot', 'ordinal', 'uniform'
                (ordinal scaled 0-1). Defaults to 'onehot'.

        Returns:
            pd.DataFrame: A new DataFrame with encoded
                categorical columns.
        """

        cat_df = self.df.select_dtypes(
            exclude=np.number
        )

        if len(cat_df.columns) == 0:
            return pd.DataFrame()

        if method == 'onehot':

            encoder = OneHotEncoder(
                sparse_output=False
            )

            encoded = encoder.fit_transform(
                cat_df
            )

            return pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out()
            )

        else:

            encoder = OrdinalEncoder()

            encoded = encoder.fit_transform(
                cat_df
            )

            encoded_df = pd.DataFrame(
                encoded,
                columns=cat_df.columns
            )

            if method == 'uniform':

                scaler = MinMaxScaler()

                encoded_df[:] = scaler.fit_transform(
                    encoded_df
                )

            return encoded_df

    def merge_normalized_data(
            self,
            num_method='standard',
            cat_method='onehot'
    ):
        """
        Creates a unified DataFrame by merging normalized
        numeric data with encoded categorical data.

        Parameters:
            num_method (str): Scaling method for numeric
                columns. Defaults to 'standard'.
            cat_method (str): Encoding method for categorical
                columns. Defaults to 'onehot'.

        Returns:
            pd.DataFrame: A combined DataFrame of scaled
                numeric and encoded categorical data.
        """

        num = self.extract_normalized_numeric_data(
            num_method
        )

        cat = self.extract_normalized_categorical_data(
            cat_method
        )

        return pd.concat(
            [num, cat],
            axis=1
        )

    def plot_numeric_distribution(
            self,
            column
    ):
        """
        Generates a 3-panel subplot for a numeric column
        containing a Violin/Box plot, a Scatter plot
        (Index vs Value), and a Histogram.

        Parameters:
            column (str): The numeric column to visualize.
        """

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Violin/Box",
                "Scatter",
                "Histogram"
            ]
        )

        fig.add_trace(
            go.Violin(
                y=self.df[column]
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                y=self.df[column],
                mode='markers'
            ),
            row=1,
            col=2
        )

        fig.add_trace(
            go.Histogram(
                x=self.df[column]
            ),
            row=1,
            col=3
        )

        fig.show()

    def plot_relationship(
            self,
            col1,
            col2
    ):
        """
        Detects column types and automatically selects
        the appropriate chart for visualizing the
        relationship between two columns.

        - Numeric-Numeric: Scatter with OLS trendline.
        - Categorical-Numeric: Box plot with all points.
        - Categorical-Categorical: Grouped bar chart.

        Parameters:
            col1 (str): First column name.
            col2 (str): Second column name.
        """

        is_num1 = pd.api.types.is_numeric_dtype(
            self.df[col1]
        )

        is_num2 = pd.api.types.is_numeric_dtype(
            self.df[col2]
        )

        if is_num1 and is_num2:

            fig = px.scatter(
                self.df,
                x=col1,
                y=col2,
                trendline='ols'
            )

        elif is_num1 != is_num2:

            if is_num1:
                cat = col2
                num = col1
            else:
                cat = col1
                num = col2

            fig = px.box(
                self.df,
                x=cat,
                y=num,
                points='all'
            )

        else:

            temp = pd.crosstab(
                self.df[col1],
                self.df[col2]
            )

            fig = px.bar(
                temp
            )

        fig.show()

    def plot_categorical_frequency(
            self,
            column
    ):
        """
        Creates a bar chart for a categorical column
        displaying both raw counts and percentage labels.

        Parameters:
            column (str): The categorical column to
                visualize.
        """

        counts = self.df[column].value_counts()

        percent = (
            counts /
            counts.sum()
            * 100
        ).round(2)

        fig = px.bar(
            x=counts.index,
            y=counts.values,
            text=percent.astype(str)+"%"
        )

        fig.show()

    def _cramers_v(self, col1, col2):
        """
        Computes Cramer's V statistic for two categorical
        columns to measure their association strength.

        Parameters:
            col1 (str): First categorical column name.
            col2 (str): Second categorical column name.

        Returns:
            float: Cramer's V value between 0 and 1.
        """

        contingency = pd.crosstab(
            self.df[col1],
            self.df[col2]
        )

        chi2 = chi2_contingency(contingency)[0]

        n = contingency.sum().sum()

        min_dim = min(contingency.shape) - 1

        if min_dim == 0:
            return 0

        return np.sqrt(chi2 / (n * min_dim))

    def _eta_correlation(self, num_col, cat_col):
        """
        Computes the Eta correlation ratio (via ANOVA)
        between a numeric and a categorical column to
        measure their association strength.

        Parameters:
            num_col (str): The numeric column name.
            cat_col (str): The categorical column name.

        Returns:
            float: Eta value between 0 and 1.
        """

        groups = [
            group.dropna().values
            for name, group
            in self.df.groupby(cat_col)[num_col]
        ]

        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return 0

        grand_mean = self.df[num_col].dropna().mean()

        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for g in groups
        )

        ss_total = sum(
            ((g - grand_mean) ** 2).sum()
            for g in groups
        )

        if ss_total == 0:
            return 0

        return np.sqrt(ss_between / ss_total)

    def plot_all_associations_heatmap(self):
        """
        Generates a unified association heatmap that
        visualizes relationships across all data types:

        - Numeric-Numeric: Pearson's r correlation.
        - Categorical-Categorical: Cramer's V statistic.
        - Mixed (Num-Cat): Eta correlation ratio (ANOVA).
        """

        num_cols = self.df.select_dtypes(
            include=np.number
        ).columns.tolist()

        cat_cols = self.df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        all_cols = num_cols + cat_cols

        n = len(all_cols)

        matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=all_cols,
            columns=all_cols
        )

        # Numeric-Numeric: Pearson r
        if len(num_cols) > 0:

            corr = self.df[num_cols].corr()

            for c1 in num_cols:
                for c2 in num_cols:
                    matrix.loc[c1, c2] = corr.loc[c1, c2]

        # Categorical-Categorical: Cramer's V
        for i, c1 in enumerate(cat_cols):
            for j, c2 in enumerate(cat_cols):

                if c1 == c2:
                    matrix.loc[c1, c2] = 1.0

                elif j > i:
                    v = self._cramers_v(c1, c2)
                    matrix.loc[c1, c2] = v
                    matrix.loc[c2, c1] = v

        # Mixed (Num-Cat): Eta correlation ratio
        for num_col in num_cols:
            for cat_col in cat_cols:

                eta = self._eta_correlation(
                    num_col,
                    cat_col
                )

                matrix.loc[num_col, cat_col] = eta
                matrix.loc[cat_col, num_col] = eta

        fig = px.imshow(
            matrix.astype(float),
            text_auto='.2f',
            title="Unified Association Heatmap",
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )

        fig.show()
