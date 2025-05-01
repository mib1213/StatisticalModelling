import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
from statsmodels.tools import add_constant

def show_missing_values(df):
    def min_or_nan(col):
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            return str(round(df[col].min(), 2))
        return np.nan
    def max_or_nan(col):
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            return str(round(df[col].max(), 2))
        return np.nan

    missing_df = pd.DataFrame({
        'S. No.': range(1, len(df.columns) + 1),
        'Column Name': df.columns,
        'Min': [min_or_nan(col) for col in df.columns],
        'Max': [max_or_nan(col) for col in df.columns],
        'n Unique': df.nunique(),
        'NaN count': df.isna().sum(),
        'NaN percentage': (df.isna().mean() * 100).round(3).astype(str) + '%',
        'dtype': df.dtypes.astype(str),

    }).set_index('S. No.')

    unique_dtypes = missing_df['dtype'].unique()
    palette = sns.color_palette("Set2", n_colors=len(unique_dtypes))
    dtype_color_map = {dt: f"background-color: {mcolors.to_hex(color)}" for dt, color in zip(unique_dtypes, palette)}

    def color_row(row):
        return [dtype_color_map.get(row['dtype'], "")] * len(row)

    return missing_df.style.apply(color_row, axis=1)

def impute_mmm(series):
    series = series.dropna()
    mean = series.mean()
    median = series.median()
    mode = series.mode().values[0]
    count = series.count()
    mean_count = (series == mean).sum()
    median_count = (series == median).sum()
    mode_count = (series == mode).sum()
    mean_perc = (mean_count / count) * 100
    median_perc = (median_count / count) * 100
    mode_perc = (mode_count / count) * 100
    lower_bound = series.quantile(0.25)
    upper_bound = series.quantile(0.75)
    iqr = upper_bound - lower_bound
    lower_outliers_mask = series < (lower_bound - 1.5 * iqr)
    upper_outliers_mask = series > (upper_bound + 1.5 * iqr)
    outliers_mask = lower_outliers_mask | upper_outliers_mask
    series_without_outliers = series[~outliers_mask]
    mean_without_outliers = series_without_outliers.mean()
    median_without_outliers = series_without_outliers.median()
    mode_without_outliers = series_without_outliers.mode().values[0]
    count_without_outliers = series_without_outliers.count()
    mean_count_without_outliers = (series_without_outliers == mean_without_outliers).sum()
    median_count_without_outliers = (series_without_outliers == median_without_outliers).sum()
    mode_count_without_outliers = (series_without_outliers == mode_without_outliers).sum()
    mean_perc_without_outliers = (mean_count_without_outliers / count_without_outliers) * 100
    median_perc_without_outliers = (median_count_without_outliers / count_without_outliers) * 100
    mode_perc_without_outliers = (mode_count_without_outliers / count_without_outliers) * 100
    return pd.DataFrame({
        'Statistics': ['Mean', 'Median', 'Mode', 'Mean without Outliers', 'Median without Outliers', 'Mode without Outliers'],
        'Value': [mean, median, mode, mean_without_outliers, median_without_outliers, mode_without_outliers],
        'Count': [mean_count, median_count, mode_count, mean_count_without_outliers, median_count_without_outliers, mode_count_without_outliers],
        'Percentage': [mean_perc, median_perc, mode_perc, mean_perc_without_outliers, median_perc_without_outliers, mode_perc_without_outliers]
    }).set_index('Statistics')

def plot_top_categories(series, n=10, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    top_categories = series.value_counts().head(n).index
    sns.countplot(x=series, order=top_categories, ax=ax)
    if title:
        ax.set_title(title)
    ax.set_ylabel('Frequency')
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom',
                    xytext=(0, 3),
                    textcoords='offset points')
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

def plot_histogram(series, gap=5, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    sns.histplot(series, bins=int(series.max() - series.min()), kde=True, ax=ax)
    if title:
        ax.set_title(title)
    mean = series.mean()
    median = series.median()
    ax.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'{mean = :.2f}')
    ax.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'{median = :.2f}')
    ax.set_xticks(range(int(series.min()), int(series.max()), gap))
    ax.legend()

def show_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

def show_mar_relation(df, target_col, top_n=10, gap=5):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in DataFrame")
    if df[target_col].isna().sum() == 0:
        return f'No missing values in {target_col}'
    not_missing = df[df[target_col].notnull()]
    missing = df[df[target_col].isnull()]
    for col in df.columns:
        if col == target_col:
            continue 
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            plot_func = plot_histogram
        else:
            plot_func = plot_top_categories 
        plot_func(not_missing[col].dropna(), ax=axes[0])
        axes[0].set_title(f"{col} (target not missing)")
        plot_func(missing[col].dropna(), ax=axes[1])
        axes[1].set_title(f"{col} (target missing)")
        plt.tight_layout()

def impute_random(series, random_seed=None):
    imputed_series = series.copy()
    missing_mask = imputed_series.isna()
    n_missing = missing_mask.sum()
    if n_missing == 0:
        return imputed_series
    imputed_values = series.dropna().sample(n_missing, replace=True, random_state=random_seed)
    imputed_series[missing_mask] = imputed_values.values
    return imputed_series

def convert_to_categorical(df, columns):
    df_ = df.copy()
    for col in columns:
        assert col in df.columns, f"Column {col} is not present in df"
        df_[col] = df_[col].astype('object')
    return df_

def get_column_name_mapping(file_path):
    return pd.read_csv(file_path, index_col='old').new.to_dict()

def create_column_name_mapping(df, file_path):
    df = pd.DataFrame({'old': df.columns, 'new': '', 'is_categorical': ''})
    df.to_csv(file_path, index=False)

def get_categorical_columns(file_path):
    df = pd.read_csv(file_path).dropna().astype({'is_categorical': bool})
    return df[df.is_categorical].new.tolist()

def do_basic_cleaning(df):
    df_ = df.copy().drop_duplicates()
    new_column_names = get_column_name_mapping('column_name_mapping.csv')
    df_ = df_.rename(columns=new_column_names)
    categorical_columns = get_categorical_columns('column_name_mapping.csv')
    df_ = convert_to_categorical(df_, categorical_columns)
    return df_

def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, p, df, expected = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def group_rare_categories(series, threshold=0.03):
    category_counts = series.value_counts(normalize=True)
    cumulative_distribution = category_counts.cumsum()
    rare_categories = category_counts[cumulative_distribution > (1 - threshold)].index
    return series.replace(rare_categories, 'Other')

def binning(series, bins=4):
    nunique = series.nunique()
    if nunique <= bins:
        return series.astype('category')
    return pd.qcut(series, bins, duplicates='drop').astype('category')

def cramers_v_matrix(df):
    df_ = df.dropna().drop_duplicates()
    df_cat = df_.select_dtypes(include=['object', 'category', 'bool']).copy()
    df_num = df_.select_dtypes(include=['number']).copy()
    matrix = pd.DataFrame(index=df_.columns, columns=df_.columns, dtype=float)
    bool_cols = df_.select_dtypes(include=['bool']).columns
    df_[bool_cols] = df_[bool_cols].astype('category')
    not_bool_cols = df_.columns.difference(bool_cols)

    for col in not_bool_cols:
        df_[col] = group_rare_categories(df_[col], threshold=0.03)

    if not df_num.empty:
        df_num = df_num.apply(binning, bins=4)
    
    df_combined = pd.concat([df_cat, df_num], axis=1)
    matrix = pd.DataFrame(index=df_combined.columns, columns=df_combined.columns, dtype=float)

    for i, col1 in enumerate(df_combined.columns):
        for j, col2 in enumerate(df_combined.columns):
            if i == j:
                matrix.loc[col1, col2] = 1.0
            elif i > j:
                 matrix.loc[col1, col2] = cramers_v(df_combined[col1], df_combined[col2])
            else:
                matrix.loc[col1, col2] = np.nan
    return matrix

def correlation_heatmap(matrix, title=None, cmap='Oranges', min=None):
    num_vars = len(matrix)    
    fig_width = max(8, num_vars * 0.6)
    fig_height = max(6, num_vars * 0.6)   
    plt.figure(figsize=(fig_width, fig_height))  
    ax = sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        linewidths=1, 
        linecolor="white", 
        annot_kws={"size": max(8, 15 - num_vars * 0.5)},
        cbar_kws={"shrink": 0.5, "aspect": 20},
        square=True,
        mask=np.triu(np.ones_like(matrix, dtype=bool), k=1),
        vmin=min
    )  
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_facecolor('white')
    return ax

def cat_corr_heatmap(matrix):
    # accessible_orange = '#CF4B00'
    # healthy_orange = '#EC6602'
    # healthy_orange_50 = '#F9B591'
    # healthy_orange_25 = '#FDDDCB'
    # neutral_cream = '#FFF4E6'
    # custom_cmap = LinearSegmentedColormap.from_list(
    #     None, 
    #     [neutral_cream, healthy_orange_25, healthy_orange, accessible_orange],
    #     N=256
    # )
    ax = correlation_heatmap(matrix, title="Cramers V", cmap='Blues', min=0)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "0.5", "1"])
    plt.show()

def num_corr_heatmap(matrix):
    # accessible_orange = '#CF4B00'
    # healthy_orange = '#EC6602'
    # healthy_orange_50 = '#F9B591'
    # healthy_orange_25 = '#FDDDCB'
    # neutral_cream = '#FFF4E6'
    # siemens_petrol_25 = '#C8E6E6'
    # siemens_petrol_50 = '#87D2D2'
    # siemens_petrol = '#009999'
    # sh_black_10 = '#E6E6E6'
    # custom_cmap = LinearSegmentedColormap.from_list(
    #     None, 
    #     [siemens_petrol, siemens_petrol_50, siemens_petrol_25, sh_black_10, healthy_orange_25, healthy_orange_50, healthy_orange],
    #     N=256
    # )
    ax = correlation_heatmap(matrix, title="Pearson", cmap='coolwarm', min=-1)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["-1", "0", "1"])
    plt.show()

def plot_bar_against(df, with_col, num_cols, hue=None, title=None):
    """
    Mean bar plot for a single categorical column against all numerical columns.
    """
    ncols = 5
    nrows = (len(num_cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))

    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.barplot(x=with_col, y=col, data=df, errorbar='sd', ax=axes[i], estimator='mean', hue=hue)
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f'Mean by {with_col}')

    plt.tight_layout()
    plt.show()

def check_normality(series):
    """
    Check normality of a series using four different tests along with a Q-Q plot and a histogram.
    """
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    stats.probplot(series, plot=ax[0], rvalue=True);
    ax[0].set_title('Q-Q Plot')
    sns.histplot(series, bins=30, kde=True, ax=ax[1]);
    ax[1].set_title('Histogram with KDE')
    plt.show()
    _, p = stats.shapiro(series)
    print(f"Shapiro-Wilk normality test: p-value = {p}")
    _, p = stats.normaltest(series)
    print(f"D'Agostino's K^2 normality test: p-value = {p}")
    _, p = stats.kstest((series - series.mean()) / series.std(), 'norm')
    print(f"Kolmogorov-Smirnov normality test: p-value = {p}")
    anderson = stats.anderson(series)
    print(f"Anderson-Darling normality test: statistic = {anderson.statistic}, critical value = {anderson.critical_values[2]}")

def check_homogeneity(df, groupby, column):
    sns.boxplot(data=df, x=groupby, y=column)
    plt.show()
    cats = [df[df[groupby] == cat][column] for cat in df[groupby].unique()]
    levene = stats.levene(*cats, center='mean')
    print(f"Levene test statistic: {levene.statistic}, p-value = {levene.pvalue}")
    brown_forsythe = stats.levene(*cats, center='median')
    print(f"Brown-Forsythe test statistic: {brown_forsythe.statistic}, p-value = {brown_forsythe.pvalue}")
    bartlett = stats.bartlett(*cats)
    print(f"Bartlett test statistic: {bartlett.statistic}, p-value = {bartlett.pvalue}")

def plot_strip_against(df, with_col, num_cols, hue=None, title=None):
    """
    Strip plot for a single categorical column against all numerical columns.
    """
    ncols = 5
    nrows = (len(num_cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))

    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.stripplot(x=with_col, y=col, data=df, ax=ax, hue=hue, alpha=0.8, dodge=True)
        ax.legend_.remove()
    if title:
        fig.suptitle(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_line_against(df, with_col, num_cols, title=None):
    """
    Line plot for a single categorical column against all numerical columns.
    """
    ncols = 5
    nrows = (len(num_cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))

    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.lineplot(x=with_col, y=col, data=df, ax=ax, errorbar='sd')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_point_against(df, with_col, num_cols, hue=None, title=None):
    """
    Point plot for a single categorical column against all numerical columns.
    """
    ncols = 5
    nrows = (len(num_cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))

    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.pointplot(x=with_col, y=col, data=df, ax=ax, errorbar='sd', hue=hue)
        if i != 0:
            ax.legend_.remove()
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_regression_results(X, y, results):
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Regression plot
    sns.regplot(x=X, y=y, ax=ax[0])
    ax[0].set_title("Regression Plot")

    # Plot 2: Residuals
    sns.residplot(x=X, y=y, lowess=True, ax=ax[1])
    ax[1].set_title("Residuals Plot")
    ax[1].set_ylabel("Residuals")

    # Plot 3: Predicted vs Actual
    y_pred = results.predict(add_constant(X))
    sns.scatterplot(x=y, y=y_pred, ax=ax[2])
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax[2].plot(lims, lims, '--', color='b')  # Reference line
    ax[2].set_title("Predicted vs Actual")
    ax[2].set_xlabel("Actual")
    ax[2].set_ylabel("Predicted")

    plt.tight_layout()
    plt.show()