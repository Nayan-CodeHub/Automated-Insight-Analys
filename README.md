# 📊 Automated Insight Analyst

An end-to-end AI/ML-style analytics dashboard for the Datathon problem statement:

> Automatically turn a raw spreadsheet-style dataset into cleaned data, analysis, visualizations and plain-language insights.

## Features

- CSV / Excel upload
- Automatic column-type detection:
  - Numeric
  - Categorical
  - Date
- Automatic cleaning:
  - Duplicate removal
  - Missing numeric values → median
  - Missing categorical values → mode / Unknown
  - Date parsing
  - String formatting cleanup
- Analytics:
  - Correlation analysis
  - IQR outlier detection
  - K-Means clustering
- Automatic chart selection
- Interactive Plotly dashboard
- Plain-language findings
- Local AI data assistant with dataset questions
- Optional OpenAI executive summary integration
- Download cleaned dataset
- Works with unseen datasets without changing column names in code

## Run in VS Code

### 1. Open the project folder

Open `automated_insight_analyst` in VS Code.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
venv\Scripts\activate.bat
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the dashboard

```powershell
python -m streamlit run app.py
```

The browser will open the dashboard.

### Optional AI integration

The dashboard works without an API key. To enable AI-generated executive summaries, set an OpenAI key before starting Streamlit:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python -m streamlit run app.py
```

You can also enter the key in the sidebar under **AI integration**. Only calculated dataset statistics, cleaning actions, and generated insights are sent to the provider; the uploaded file itself is not sent.

## Project structure

```text
automated_insight_analyst/
│
├── app.py                 # Streamlit UI
├── analyst.py             # Detection, cleaning, ML and insights
├── charts.py              # Automatic Plotly charts
├── requirements.txt
├── README.md
└── sample_data/
    └── sales_demo.csv
```

## Datathon demo flow

1. Start with the included sales dataset.
2. Show automatic type detection.
3. Show duplicate/missing-value handling.
4. Open the Dashboard tab.
5. Explain that charts were selected from detected column types.
6. Open ML & Insights and show correlations, outliers and clustering.
7. Upload a different CSV with different column names.
8. Show that the same pipeline adapts without rewriting the program.

## Important judging point

The system deliberately avoids hard-coding names such as `Sales`, `Region`, or `Category`. It detects the structure of the uploaded dataset and builds the analysis around the available columns.

## Future upgrades

For an advanced version, add:
- Automatic target-column detection
- More clustering algorithms
- Forecasting
- Dataset anomaly scoring
- PDF report export
- User-defined filters
- Database/API input
- Model explainability
