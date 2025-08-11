# 💧 Global Drinking Water Access Explorer

An interactive Streamlit-based Exploratory Data Analysis (EDA) dashboard for global (or multi-country) drinking water access data. Upload a CSV containing a **Year** (or *Years*) column plus one column per country (numeric percentages). Interactively explore trends, comparisons, correlations, changes, and distributions.

## ✨ Features
- Multi-section selectable EDA interface (enable/disable modules from sidebar)
- Automatic Year column detection / creation
- Robust Year sanitization (handles non-numeric / inf / missing)
- Section 1: Single-country Bar Graph (year-over-year)
- Section 2: Multi-country Line Graph (optional 3-year rolling mean)
- Section 3: Correlation Heatmap (Pearson / Spearman / Kendall)
- Section 4: All-country Scatter (Plotly zoom/pan or Altair fallback)
- Bar Comparison (latest year / average / specific year)
- Change Analysis (First vs Last, Abs & % Change, Years Span)
- Distribution (Histogram + KDE)
- Pairwise Scatter (sampled first N years)
- Missing Values summary + heatmap
- Filtered CSV download
- Optional raw data preview

## 📁 Expected Data Format
CSV with columns like:
```
Year,CountryA,CountryB,CountryC
2000,75.2,63.1,80.0
2001,76.0,64.0,81.5
...
```
If the file has a column named `Years`, it will be renamed automatically to `Year`.
If no `Year` column exists, an index-based Year (1..N) will be inserted.

All non-Year numeric columns are treated as country metrics (percentage access).

## 🚀 Quick Start
1. Create & activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your data file as `water_access_data.csv` in the project root (or use the upload widget in-app).
4. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```
5. Open the provided local URL (usually http://localhost:8501) in your browser.

## 🧩 Sidebar Controls
- Upload CSV (optional if file exists locally)
- Country multiselect (type-ahead search)
- Quick-add dropdown to append one more country fast
- Toggle raw data display
- Enable/disable EDA sections

## 🔍 EDA Sections (Toggle On/Off)
| Section | Description |
|---------|-------------|
| Overview | Dataset shape, country count, year range, overall mean |
| Summary Stats | Descriptive statistics (per selected country) |
| 1. Bar Graph for a Country | Single-country annual bar chart |
| 2. Line Graph for Selected Countries | Multi-country time series (optional rolling mean) |
| Bar Comparison | Latest / Average / Chosen year comparison |
| 3. Heatmap for Selected Countries | Correlation matrix with selectable method |
| Change Analysis | First vs Last, absolute & percent change, duration |
| Distribution | Histogram + KDE for a selected country |
| Pairwise Scatter | Sampled scatter across first N years |
| 4. Scatter Plot for All Countries | All values: Country vs % colored by Year (Plotly zoom or Altair) |
| Missing Values | Counts + heatmap |

## 🛠 Implementation Notes
- Plotly is optional but installed by default for superior zooming in Section 4.
- Altair used for most interactive charts (lightweight & declarative).
- Seaborn / Matplotlib used for heatmaps, histograms, and fallback visuals.
- Data melt helpers convert wide country columns into long format for uniform plotting.
- Year labels protected from cutoff (extra bottom margin / axis padding).
- Change Analysis excludes CAGR (simplified per design decision).

## ⚠️ Data Quality Handling
- Non-numeric Years coerced to NaN; invalid rows dropped (with a warning).
- Infinite values in Year replaced with NaN before dropping.
- Year cast to int only if all remaining are whole numbers.
- Empty / all-NaN segments in change stats gracefully skipped.

## 📦 Dependencies
See `requirements.txt` (key libraries: streamlit, pandas, numpy, seaborn, altair, matplotlib, plotly).

## 🧪 Extensibility Ideas
Potential future enhancements:
- Forecasting (Prophet / statsmodels)
- Clustering countries by trajectories
- PCA dimensionality reduction & similarity mapping
- Anomaly detection (e.g., isolation forest on yearly deltas)
- Geographic mapping layer (e.g., pydeck / folium / geojson)
- Image export buttons for charts

## 🔧 Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| App warns about invalid Year rows | Non-numeric / missing year data | Clean source CSV or accept filtered dataset |
| No Plotly zoom toggle | plotly not installed / import failed | Reinstall: `pip install plotly` |
| Empty charts | No countries selected / all NaN | Select countries; verify numeric values |
| Percent change NaN | First value 0 or missing | Data artifact; consider filtering |



## 🙌 Acknowledgements
Built with Streamlit • Altair • Seaborn • Matplotlib • Plotly.
