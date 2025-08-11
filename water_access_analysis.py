import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from io import BytesIO

# Try optional Plotly for enhanced zoom
try:
    import plotly.express as px  # type: ignore
    _plotly_available = True
except Exception:
    _plotly_available = False

st.set_page_config(page_title="Water Access EDA", layout="wide", page_icon="💧")
st.title("💧 Global Drinking Water Access Explorer")
st.markdown("Perform interactive exploratory data analysis (EDA) on water access data.")

# ------------------ Data Loading ------------------ #
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Try to detect a year-like column
    year_col_candidates = [c for c in df.columns if c.lower() in ("year", "years")]
    if year_col_candidates:
        df = df.rename(columns={year_col_candidates[0]: "Year"})
    else:
        # If no Year column, attempt index
        if not any(c.lower()=="year" for c in df.columns):
            df.insert(0, "Year", range(1, len(df)+1))
    return df

uploaded = st.sidebar.file_uploader("Upload water_access_data.csv", type=["csv"])  # optional
if uploaded:
    df = load_data(uploaded)
else:
    try:
        df = load_data("water_access_data.csv")
    except Exception as e:
        st.warning("Please upload a CSV file containing a 'Years' or 'Year' column and country columns.")
        st.stop()

# Standardize Year column name
if 'Years' in df.columns and 'Year' not in df.columns:
    df = df.rename(columns={'Years': 'Year'})

# Clean Year column (handle NaN/inf and non-numeric)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
# Replace inf values (if any) with NaN then drop
if np.isinf(df['Year']).any():
    df.loc[np.isinf(df['Year']), 'Year'] = np.nan
invalid_year_count = df['Year'].isna().sum()
if invalid_year_count:
    st.warning(f"{invalid_year_count} row(s) have invalid Year values and were excluded from analysis.")
    df = df.dropna(subset=['Year']).reset_index(drop=True)
# Ensure Year sorted and integer-like (keep as int if all whole numbers)
if (df['Year'] % 1 == 0).all():
    df['Year'] = df['Year'].astype(int)

# Identify numeric country columns (exclude Year)
value_cols = [c for c in df.columns if c != 'Year' and pd.api.types.is_numeric_dtype(df[c])]

# Sidebar controls
st.sidebar.header("Controls")
# Enhanced multiselect (Streamlit provides type-ahead search automatically)
selected_countries = st.sidebar.multiselect(
    "Select Countries (type to search)", value_cols,
    default=value_cols[: min(5, len(value_cols))]
)
# Optional quick single-country add (dropdown) for usability
quick_add = st.sidebar.selectbox("Quick add a country", ["(None)"] + sorted([c for c in value_cols if c not in selected_countries]))
if quick_add != "(None)":
    selected_countries = selected_countries + [quick_add]

# Remove potential duplicates while preserving order
seen = set()
selected_countries = [c for c in selected_countries if not (c in seen or seen.add(c))]

show_raw = st.sidebar.checkbox("Show Raw Data", False)

eda_sections = st.sidebar.multiselect(
    "Select EDA Sections",
    [
        "Overview", "Summary Stats",
        "1. Bar Graph for a Country",
        "2. Line Graph for Selected Countries",
        "Bar Comparison",
        "3. Heatmap for Selected Countries",
        "Change Analysis", "Distribution", "Pairwise Scatter",
        "4. Scatter Plot for All Countries", "Missing Values"
    ],
    default=["Overview", "1. Bar Graph for a Country", "2. Line Graph for Selected Countries", "3. Heatmap for Selected Countries", "4. Scatter Plot for All Countries", "Change Analysis"]
)

# Helper melt
@st.cache_data(show_spinner=False)
def melt_data(df: pd.DataFrame, cols):
    if not cols:
        return pd.DataFrame(columns=['Year', 'Country', 'Percentage'])
    return df.melt(id_vars='Year', value_vars=cols, var_name='Country', value_name='Percentage')

melted = melt_data(df, selected_countries)

# ------------------ Overview ------------------ #
if "Overview" in eda_sections:
    st.subheader("📁 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{df.shape[0]}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Countries", f"{len(value_cols)}")
    year_min, year_max = df['Year'].min(), df['Year'].max()
    c4.metric("Year Range", f"{year_min} – {year_max}")
    overall_mean = df[value_cols].stack().mean() if value_cols else np.nan
    c5.metric("Overall Mean %", f"{overall_mean:.2f}" if not np.isnan(overall_mean) else "-")
    if show_raw:
        st.dataframe(df, use_container_width=True)

# ------------------ Summary Stats ------------------ #
if "Summary Stats" in eda_sections and selected_countries:
    st.subheader("📊 Summary Statistics")
    st.dataframe(df[selected_countries].describe().T, use_container_width=True)

# ------------------ 1. Bar Graph for a Country ------------------ #
if "1. Bar Graph for a Country" in eda_sections and value_cols:
    st.subheader("1. Bar Graph for a Country")
    country_single = st.selectbox("Select a Country", value_cols, key="single_bar_country")
    rotate_years = st.checkbox("Rotate Year Labels (45°)", value=True, key="rotate_years_single")
    # Prepare clean year series
    year_series = pd.to_numeric(df['Year'], errors='coerce')
    valid_mask = year_series.notna() & ~np.isinf(year_series)
    if not valid_mask.any():
        st.info("No valid Year data available to plot.")
    else:
        years_vals = year_series[valid_mask].astype(int)
        values = df.loc[valid_mask, country_single]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(years_vals, values, color='purple', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage')
        ax.set_title(f"% Access to Drinking Water in {country_single} Over Years")
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(years_vals)
        ax.set_xticklabels(years_vals, rotation=45 if rotate_years else 0, ha='right' if rotate_years else 'center')
        plt.tight_layout()
        st.pyplot(fig)

# ------------------ Line Trends (renamed #2) ------------------ #
if "2. Line Graph for Selected Countries" in eda_sections and selected_countries:
    st.subheader("2. Line Graph for Selected Countries")
    smooth = st.checkbox("Apply Rolling Mean (window=3)", value=False, key="smooth_line2")
    chart_df = melted.copy()
    if smooth and not chart_df.empty:
        chart_df['Smoothed'] = chart_df.groupby('Country')['Percentage'].transform(lambda s: s.rolling(3, min_periods=1).mean())
        y_field2 = 'Smoothed'
    else:
        y_field2 = 'Percentage'
    if chart_df.empty:
        st.info("Select at least one country to see the time series.")
    else:
        line_chart2 = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X('Year:O', title='Year'),
                y=alt.Y(f'{y_field2}:Q', title='Percentage'),
                color=alt.Color('Country:N', legend=alt.Legend(title="Country")),
                tooltip=['Country', 'Year', alt.Tooltip(y_field2, format='.2f')]
            )
            .properties(height=400)
            .interactive()
        )
        st.altair_chart(line_chart2, use_container_width=True)

# ------------------ Bar Comparison ------------------ #
if "Bar Comparison" in eda_sections and selected_countries:
    st.subheader("📊 Bar Comparison")
    mode = st.radio("Bar Mode", ["Latest Year", "Average", "Select Year"], horizontal=True)
    if mode == "Latest Year":
        y_sel = df['Year'].max()
        bar_df = df[df['Year'] == y_sel][['Year'] + selected_countries].melt('Year', var_name='Country', value_name='Percentage')
        subtitle = f"Latest Year: {y_sel}"
    elif mode == "Average":
        bar_df = pd.DataFrame({'Country': selected_countries, 'Percentage': [df[c].mean() for c in selected_countries]})
        subtitle = "Average Across Years"
    else:
        year_choice = st.select_slider("Select Year", options=sorted(df['Year'].unique()))
        bar_df = df[df['Year'] == year_choice][['Year'] + selected_countries].melt('Year', var_name='Country', value_name='Percentage')
        subtitle = f"Year: {year_choice}"

    bar_chart = (
        alt.Chart(bar_df)
        .mark_bar()
        .encode(
            x=alt.X('Country:N', sort='-y'),
            y=alt.Y('Percentage:Q'),
            color=alt.Color('Country:N', legend=None),
            tooltip=['Country', alt.Tooltip('Percentage:Q', format='.2f')]
        )
        .properties(height=400, title=subtitle)
    )
    st.altair_chart(bar_chart, use_container_width=True)

# ------------------ 3. Heatmap for Selected Countries ------------------ #
if "3. Heatmap for Selected Countries" in eda_sections and selected_countries:
    st.subheader("3. Heatmap for Selected Countries")
    if len(selected_countries) < 2:
        st.info("Select at least two countries for correlation analysis.")
    else:
        corr_method3 = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"], index=0, key="corr_method3")
        corr3 = df[selected_countries].corr(method=corr_method3)
        fig3, ax3 = plt.subplots(figsize=(min(10, 1 + len(selected_countries)), 0.6 * len(selected_countries)))
        sns.heatmap(corr3, annot=True, fmt='.2f', cmap='viridis', linewidths=.5, ax=ax3)
        ax3.set_title(f"Correlation ({corr_method3.title()})")
        st.pyplot(fig3)

# ------------------ Change Analysis ------------------ #
if "Change Analysis" in eda_sections and selected_countries:
    st.subheader("📉 Change & Growth Analysis (CAGR removed)")
    change_rows = []
    for c in selected_countries:
        series = df[["Year", c]].dropna()
        if series.empty:
            continue
        first, last = series.iloc[0, 1], series.iloc[-1, 1]
        pct_change = ((last - first) / first * 100) if first not in (0, np.nan) else np.nan
        years_span = series['Year'].iloc[-1] - series['Year'].iloc[0]
        change_rows.append({
            'Country': c,
            'First (%)': first,
            'Last (%)': last,
            'Abs Change': last - first,
            '% Change': pct_change,
            'Years Span': years_span
        })
    change_df = pd.DataFrame(change_rows)
    if change_df.empty:
        st.info("No sufficient data to compute change metrics.")
    else:
        st.dataframe(
            change_df.style.format({'First (%)': '{:.2f}', 'Last (%)': '{:.2f}', 'Abs Change': '{:.2f}', '% Change': '{:.2f}', 'Years Span': '{:.0f}'}),
            use_container_width=True
        )
        valid_pct = change_df.dropna(subset=['% Change'])
        if not valid_pct.empty:
            best = valid_pct.loc[valid_pct['% Change'].idxmax()]
            worst = valid_pct.loc[valid_pct['% Change'].idxmin()]
            c1, c2 = st.columns(2)
            c1.success(f"Highest % Change: {best['Country']} ({best['% Change']:.2f}%)")
            c2.error(f"Lowest % Change: {worst['Country']} ({worst['% Change']:.2f}%)")

# ------------------ Distribution ------------------ #
if "Distribution" in eda_sections and selected_countries:
    st.subheader("📦 Distribution (Histogram / KDE)")
    country_dist = st.selectbox("Select Country", selected_countries, key="dist_country")
    bins = st.slider("Bins", 5, 50, 15, key="bins")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[country_dist].dropna(), bins=bins, kde=True, ax=ax, color='teal')
    ax.set_title(f"Distribution of {country_dist}")
    st.pyplot(fig)

# ------------------ Pairwise Scatter ------------------ #
if "Pairwise Scatter" in eda_sections and len(selected_countries) >= 2:
    st.subheader("🔍 Pairwise Scatter (Sampled)")
    sample_years = st.slider("Sample first N years", 5, len(df), min(20, len(df)), key="sample_years")
    pair_df = df[['Year'] + selected_countries].head(sample_years)
    melted_pair = pair_df.melt('Year', var_name='Country', value_name='Percentage')
    scatter = (
        alt.Chart(melted_pair)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X('Year:O'),
            y='Percentage:Q',
            color='Country:N',
            tooltip=['Country', 'Year', alt.Tooltip('Percentage:Q', format='.2f')]
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

# ------------------ 4. Scatter Plot for All Countries ------------------ #
if "4. Scatter Plot for All Countries" in eda_sections and value_cols:
    st.subheader("4. Scatter Plot for All Countries")
    # Option toggles (removed dense seaborn option)
    colA, colB = st.columns([1,1])
    with colA:
        rotate_labels = st.checkbox("Rotate X Labels (45°)", value=True, key="rotate_scatter_all")
    with colB:
        use_plotly = st.checkbox("Plotly Zoom Mode", value=_plotly_available, disabled=not _plotly_available, help="Uses Plotly for full zoom & pan if installed.")

    melted_all = df.melt(id_vars='Year', value_vars=value_cols, var_name='Country', value_name='Percentage')

    if use_plotly and _plotly_available:
        fig_pl = px.scatter(
            melted_all,
            x='Country', y='Percentage', color='Year',
            hover_data=['Country','Year','Percentage'],
            title='Interactive Scatter (Zoom / Pan)',
            height=700,
        )
        if rotate_labels:
            fig_pl.update_layout(xaxis_tickangle=-45)
        # Increase bottom margin to avoid cutoff & enable automargin
        fig_pl.update_layout(
            legend_title_text='Year',
            margin=dict(l=10, r=10, t=40, b=160)
        )
        fig_pl.update_xaxes(automargin=True)
        st.plotly_chart(fig_pl, use_container_width=True, theme=None)
    else:
        # Interactive Altair scatter with improved axis label handling
        angle = -45 if rotate_labels else 0
        scatter_all = (
            alt.Chart(melted_all)
            .mark_circle(size=70, opacity=0.65)
            .encode(
                x=alt.X('Country:N', sort=alt.SortField('Country'), axis=alt.Axis(labelAngle=angle, labelPadding=8, labelLimit=2000)),
                y='Percentage:Q',
                color=alt.Color('Year:O', legend=alt.Legend(title='Year')),
                tooltip=['Country', 'Year', alt.Tooltip('Percentage:Q', format='.2f')]
            )
            .properties(height=700, padding={'bottom': 140})
            .interactive()
        )
        st.altair_chart(scatter_all, use_container_width=True)

# ------------------ Missing Values ------------------ #
if "Missing Values" in eda_sections:
    st.subheader("🚨 Missing Values")
    mv = df.isna().sum()
    if mv.sum() == 0:
        st.info("No missing values detected.")
    else:
        mv_df = mv.reset_index().rename(columns={'index': 'Column', 0: 'Missing'})
        mv_df.columns = ['Column', 'Missing']
        st.dataframe(mv_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(6,3))
        sns.heatmap(df.isna(), cbar=False, yticklabels=False, ax=ax)
        ax.set_title('Missing Value Map')
        st.pyplot(fig)

# ------------------ Download Filtered Data ------------------ #
if selected_countries:
    st.subheader("⬇️ Download Filtered Data")
    def to_csv(df_download):
        return df_download.to_csv(index=False).encode('utf-8')
    filt_df = df[['Year'] + selected_countries]
    st.download_button(
        label="Download CSV",
        data=to_csv(filt_df),
        file_name='filtered_water_access.csv',
        mime='text/csv'
    )

st.caption("Built with Streamlit • Altair • Seaborn • Matplotlib")


