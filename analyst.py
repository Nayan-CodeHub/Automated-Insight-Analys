import numpy as np
import pandas as pd
import json
import urllib.error
import urllib.request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def _numeric_values(series):
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"[$€£₹]", "", regex=True)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")

def _date_values(series):
    return pd.to_datetime(series, errors="coerce", format="mixed")

def detect_column_types(df):
    types = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            types[col] = "categorical"
            continue
        if pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
            continue

        non_null = s.dropna()
        if len(non_null):
            numeric_ratio = _numeric_values(non_null).notna().mean()
            if numeric_ratio >= 0.8:
                types[col] = "numeric"
                continue

            date_ratio = _date_values(non_null).notna().mean()
            if date_ratio >= 0.8:
                types[col] = "date"
                continue

        types[col] = "categorical"
    return types

def clean_data(
    df,
    types,
    remove_duplicates=True,
    numeric_fill="median",
    categorical_fill="mode",
    date_fill="median",
):
    out = df.copy()
    log = []

    if remove_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        if len(out) < before:
            log.append(f"Removed {before-len(out):,} duplicate rows.")
    else:
        log.append("Duplicate rows were preserved by the selected cleaning options.")

    for col, kind in types.items():
        if kind == "date":
            out[col] = _date_values(out[col])
            miss = int(out[col].isna().sum())
            if miss and date_fill == "median":
                fill = out[col].dropna().median()
                out[col] = out[col].fillna(fill)
                log.append(f"{col}: parsed as date and filled {miss:,} missing values with the median date.")
            elif miss:
                log.append(f"{col}: parsed as date and left {miss:,} missing values unchanged.")

        elif kind == "numeric":
            out[col] = _numeric_values(out[col])
            miss = int(out[col].isna().sum())
            if miss and numeric_fill == "median":
                fill = out[col].median()
                out[col] = out[col].fillna(fill)
                log.append(f"{col}: filled {miss:,} missing numeric values with median ({fill:.2f}).")
            elif miss and numeric_fill == "mean":
                fill = out[col].mean()
                out[col] = out[col].fillna(fill)
                log.append(f"{col}: filled {miss:,} missing numeric values with mean ({fill:.2f}).")
            elif miss:
                log.append(f"{col}: left {miss:,} missing numeric values unchanged.")

        else:
            out[col] = out[col].astype("string").str.strip()
            miss = int(out[col].isna().sum())
            if miss and categorical_fill == "mode":
                mode = out[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else "Unknown"
                out[col] = out[col].fillna(fill)
                log.append(f"{col}: filled {miss:,} missing categorical values with '{fill}'.")
            elif miss:
                out[col] = out[col].fillna("Unknown")
                log.append(f"{col}: filled {miss:,} missing categorical values with 'Unknown'.")

    return out, log

def build_cleaning_report(raw, cleaned, types):
    rows = []
    for col, kind in types.items():
        raw_missing = int(raw[col].isna().sum())
        cleaned_missing = int(cleaned[col].isna().sum())
        converted = kind in ("numeric", "date") and raw[col].dtype != cleaned[col].dtype
        rows.append({
            "Column": col,
            "Detected type": kind,
            "Missing before": raw_missing,
            "Missing after": cleaned_missing,
            "Values filled": max(0, raw_missing - cleaned_missing),
            "Type normalized": "Yes" if converted else "No",
        })
    return pd.DataFrame(rows)

def dataset_profile(df, types):
    rows = []
    for col, kind in types.items():
        s = df[col]
        rows.append({
            "Column": col,
            "Type": kind,
            "Unique": int(s.nunique(dropna=True)),
            "Missing": int(s.isna().sum()),
            "Example": str(s.dropna().iloc[0])[:60] if s.notna().any() else ""
        })
    return pd.DataFrame(rows)

def analyze_correlations(df, types):
    numeric = [c for c, t in types.items() if t == "numeric"]
    if len(numeric) < 2:
        return pd.DataFrame()
    corr = df[numeric].corr(numeric_only=True)
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i+1:]:
            value = corr.loc[a, b]
            if pd.notna(value):
                pairs.append({"Column A": a, "Column B": b, "Correlation": round(float(value), 3)})
    return pd.DataFrame(pairs).sort_values("Correlation", key=lambda x: x.abs(), ascending=False)

def detect_outliers(df, types):
    results = []
    for col, kind in types.items():
        if kind != "numeric":
            continue
        s = df[col].dropna()
        if len(s) < 5 or s.nunique() < 3:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask = (s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)
        count = int(mask.sum())
        results.append({
            "column": col,
            "count": count,
            "percentage": count / len(s) * 100
        })
    return sorted(results, key=lambda x: x["count"], reverse=True)

def perform_clustering(df, types):
    numeric = [c for c, t in types.items() if t == "numeric"]
    if len(numeric) < 2 or len(df) < 10:
        return {"available": False, "reason": "Need at least 2 numeric columns and 10 rows."}
    data = df[numeric].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10:
        return {"available": False, "reason": "Not enough complete numeric rows for clustering."}

    cols = numeric[:8]  # keeps runtime reasonable
    X = StandardScaler().fit_transform(data[cols])
    k = min(4, max(2, int(np.sqrt(len(data)/2))))
    k = min(k, 8)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    summary = data.copy()
    summary["Cluster"] = labels
    cluster_summary = summary.groupby("Cluster")[cols].mean().round(2).reset_index()
    return {"available": True, "n_clusters": k, "summary": cluster_summary}

def build_chart_specs(df, types):
    numeric = [c for c,t in types.items() if t == "numeric"]
    categorical = [c for c,t in types.items() if t == "categorical"]
    dates = [c for c,t in types.items() if t == "date"]
    specs = []

    if dates and numeric:
        specs.append({"kind":"line","x":dates[0],"y":numeric[0],
                      "title":f"{numeric[0]} over {dates[0]}"})

    for cat in categorical[:3]:
        if numeric:
            specs.append({"kind":"bar","x":cat,"y":numeric[0],
                          "title":f"{numeric[0]} by {cat}"})

    if len(numeric) >= 2:
        specs.append({"kind":"scatter","x":numeric[0],"y":numeric[1],
                      "title":f"{numeric[1]} vs {numeric[0]}"})

    if numeric:
        specs.append({"kind":"hist","x":numeric[0],"title":f"Distribution of {numeric[0]}"})
        specs.append({"kind":"box","x":numeric[0],"title":f"Outlier view: {numeric[0]}"})

    if len(numeric) >= 3:
        specs.append({"kind":"heatmap","columns":numeric[:10],
                      "title":"Numeric correlation heatmap"})

    return specs[:8]

def generate_insights(df, types, corr, outliers, clusters):
    insights = []
    numeric = [c for c,t in types.items() if t == "numeric"]
    categorical = [c for c,t in types.items() if t == "categorical"]
    dates = [c for c,t in types.items() if t == "date"]

    if dates:
        d = dates[0]
        insights.append(f"'{d}' was detected as a date column, so time-based trends can be analyzed automatically.")

    if numeric:
        col = numeric[0]
        insights.append(f"'{col}' ranges from {df[col].min():,.2f} to {df[col].max():,.2f}, with an average of {df[col].mean():,.2f}.")

    if not corr.empty:
        strongest = corr.iloc[0]
        insights.append(f"Strongest numeric relationship: {strongest['Column A']} vs {strongest['Column B']} (correlation {strongest['Correlation']:.2f}).")

    if categorical and numeric:
        cat, num = categorical[0], numeric[0]
        grouped = df.groupby(cat, dropna=False)[num].mean().sort_values(ascending=False)
        if len(grouped):
            insights.append(f"Highest average {num} occurs in '{grouped.index[0]}' for {cat}.")

    if outliers:
        top = outliers[0]
        insights.append(f"{top['column']} has {top['count']} IQR-based outliers ({top['percentage']:.1f}% of usable values).")

    if clusters.get("available"):
        insights.append(f"K-Means found {clusters['n_clusters']} groups from the numeric features, useful for segmenting similar records.")

    return insights

def answer_data_question(df, types, question):
    numeric = [column for column, kind in types.items() if kind == "numeric"]
    categorical = [column for column, kind in types.items() if kind == "categorical"]

    if question == "What is the dataset size?":
        return f"Your dataset contains {len(df):,} rows and {len(df.columns):,} columns."

    if question == "Which numeric column has the highest average?":
        if not numeric:
            return "There are no numeric columns available for this question."
        averages = df[numeric].mean().sort_values(ascending=False)
        column = averages.index[0]
        return f"{column} has the highest average value at {averages.iloc[0]:,.2f}."

    if question == "Which category appears most often?":
        if not categorical:
            return "There are no categorical columns available for this question."
        column = categorical[0]
        value = df[column].value_counts(dropna=False).index[0]
        count = int(df[column].value_counts(dropna=False).iloc[0])
        return f"The most common value is '{value}' in {column}, appearing {count:,} times."

    if question == "Where are the missing values?":
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            return "There are no missing values in the current dataset view."
        details = ", ".join(f"{column} ({count:,})" for column, count in missing.items())
        return f"Missing values were found in: {details}."

    if question == "What should I investigate first?":
        missing_total = int(df.isna().sum().sum())
        duplicate_total = int(df.duplicated().sum())
        if missing_total:
            return f"Start with missing values. There are {missing_total:,} missing cells that may affect analysis."
        if duplicate_total:
            return f"Start with duplicate records. There are {duplicate_total:,} duplicate rows in the current view."
        if numeric:
            return f"Start by exploring the distribution and outliers in '{numeric[0]}'."
        return "The dataset is clean enough to begin exploring category patterns."

    return "Choose a question to analyze the current dataset."

def generate_ai_summary(api_key, model, dataset_summary, insights, cleaning_log):
    if not api_key:
        return None, "Add an OpenAI API key to generate an AI narrative."

    prompt = {
        "dataset": dataset_summary,
        "automatic_insights": insights,
        "cleaning_actions": cleaning_log,
        "instructions": (
            "Write a concise executive summary for a business analyst. "
            "Use three headings: What stands out, Data quality, Recommended next steps. "
            "Use only the supplied facts. Do not invent causes, predictions, or numbers."
        ),
    }
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful data analyst."},
            {"role": "user", "content": json.dumps(prompt, default=str)},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip(), None
    except urllib.error.HTTPError as exc:
        return None, f"AI provider error ({exc.code}). Check the API key and model name."
    except (urllib.error.URLError, TimeoutError):
        return None, "The AI provider could not be reached. Check your internet connection."
    except (KeyError, IndexError, json.JSONDecodeError):
        return None, "The AI provider returned an unexpected response."
