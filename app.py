import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import urllib.request
import urllib.error
import json
import io
import re

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# =========================================================
# 📐 FULL-WIDTH APPLICATION
# =========================================================

st.markdown(
    """
    <style>

    /* Full width for EVERY section and EVERY tab */
    [data-testid="stMainBlockContainer"] {
        max-width: none !important;
        width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    [data-testid="stAppViewContainer"] {
        width: 100% !important;
    }

    /* Charts */
    .stPlotlyChart {
        width: 100% !important;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        width: 100% !important;
    }

    /* Tabs */
    [data-testid="stTabs"] {
        width: 100% !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# PROJECT HEADER
# =========================================================

st.markdown(
    """
    <h1 style="
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    ">
        🌱 AI-Powered Sustainable Campus Resource Advisor
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="project-subtitle">'
    'AI-driven decision support for smarter and more sustainable '
    'campus resource management'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = "data/campus_resource_data.csv"
KNOWLEDGE_PATH = "data/sustainability_knowledge.csv"


# =========================================================
# LOAD DATA
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"Unable to load campus dataset: {exc}")
    st.stop()

try:
    knowledge_df = pd.read_csv(KNOWLEDGE_PATH)
except Exception as exc:
    st.error(f"Unable to load sustainability knowledge base: {exc}")
    st.stop()


required_columns = [
    "Date",
    "Building",
    "Occupancy",
    "Temperature_C",
    "Electricity_kWh",
    "Water_Liters",
    "Waste_kg",
    "Paper_kg"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "The campus dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(subset=["Date"]).copy()


# =========================================================
# RESOURCE CONFIGURATION
# =========================================================

resources = {
    "Electricity_kWh": {
        "name": "Electricity",
        "unit": "kWh",
        "icon": "⚡",
        "action": (
            "Check HVAC systems, lighting schedules and "
            "high-load laboratory or computing equipment."
        )
    },
    "Water_Liters": {
        "name": "Water",
        "unit": "L",
        "icon": "💧",
        "action": (
            "Inspect pipelines, taps, storage tanks and "
            "possible leakage or overflow points."
        )
    },
    "Waste_kg": {
        "name": "Waste",
        "unit": "kg",
        "icon": "♻️",
        "action": (
            "Review major waste sources, improve segregation "
            "and strengthen recycling practices."
        )
    },
    "Paper_kg": {
        "name": "Paper",
        "unit": "kg",
        "icon": "📄",
        "action": (
            "Increase digital documentation, online forms "
            "and double-sided printing."
        )
    }
}

features = list(resources.keys())


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🔎 Dashboard Filters")

building_options = [
    "All Buildings"
] + sorted(
    df["Building"].dropna().unique().tolist()
)

selected_building = st.sidebar.selectbox(
    "Select Building",
    building_options
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = min_date
    end_date = max_date


view = df[
    (df["Date"].dt.date >= start_date) &
    (df["Date"].dt.date <= end_date)
].copy()

if selected_building != "All Buildings":
    view = view[
        view["Building"] == selected_building
    ].copy()


if view.empty:
    st.warning(
        "No data is available for the selected filters. "
        "Please choose another date range or building."
    )
    st.stop()


# =========================================================
# AI ANOMALY DETECTION
# =========================================================

model = IsolationForest(
    contamination=0.03,
    random_state=42
)

view["Anomaly_Label"] = model.fit_predict(
    view[features]
)

view["Anomaly"] = np.where(
    view["Anomaly_Label"] == -1,
    "Anomaly",
    "Normal"
)

alerts = view[
    view["Anomaly"] == "Anomaly"
].copy()


# =========================================================
# RESOURCE BASELINE ANALYSIS
# =========================================================

recent_window = view[
    view["Date"] >= (
        view["Date"].max() -
        pd.Timedelta(days=6)
    )
].copy()

resource_analysis = {}

increase_scores = {}

for feature in features:

    recent_avg = float(
        recent_window[feature].mean()
    )

    historical_avg = float(
        view[feature].mean()
    )

    if historical_avg != 0:
        change = (
            (recent_avg - historical_avg)
            / historical_avg
        ) * 100
    else:
        change = 0.0

    resource_analysis[feature] = {
        "recent": recent_avg,
        "historical": historical_avg,
        "change": change
    }

    increase_scores[feature] = change / 100


highest_resource = max(
    features,
    key=lambda x: resource_analysis[x]["change"]
)

highest_name = resources[
    highest_resource
]["name"]

highest_change = resource_analysis[
    highest_resource
]["change"]


# =========================================================
# BASIC SUSTAINABILITY SCORE
# =========================================================

positive_increases = [
    max(increase_scores[f], 0)
    for f in features
]

average_increase = np.mean(
    positive_increases
)

sustainability_score = int(
    np.clip(
        100 - (average_increase * 100),
        0,
        100
    )
)


# =========================================================
# ANOMALY SEVERITY
# =========================================================

total_alerts = len(alerts)

high_alerts = 0
medium_alerts = 0
low_alerts = 0

detailed_anomalies = []

for _, row in alerts.iterrows():

    building = row["Building"]

    building_data = view[
        view["Building"] == building
    ]

    resource_deviations = {}

    for resource_name, feature in {
        "Electricity": "Electricity_kWh",
        "Water": "Water_Liters",
        "Waste": "Waste_kg",
        "Paper": "Paper_kg"
    }.items():

        actual_value = float(
            row[feature]
        )

        normal_value = float(
            building_data[feature].median()
        )

        if normal_value > 0:

            deviation = (
                (actual_value - normal_value)
                / normal_value
            ) * 100

        else:

            deviation = 0.0

        resource_deviations[
            resource_name
        ] = {
            "actual": actual_value,
            "normal": normal_value,
            "deviation": deviation
        }

    main_resource = max(
        resource_deviations,
        key=lambda x: abs(
            resource_deviations[x]["deviation"]
        )
    )

    actual = resource_deviations[
        main_resource
    ]["actual"]

    normal = resource_deviations[
        main_resource
    ]["normal"]

    deviation = resource_deviations[
        main_resource
    ]["deviation"]

    if abs(deviation) >= 40:

        severity = "🔴 High"
        high_alerts += 1

    elif abs(deviation) >= 20:

        severity = "🟠 Medium"
        medium_alerts += 1

    else:

        severity = "🟡 Low"
        low_alerts += 1

    if deviation > 0:

        explanation = (
            f"{main_resource} consumption is approximately "
            f"{abs(deviation):.1f}% higher than the normal "
            f"level for {building}. The AI model identified "
            f"this as an unusual consumption pattern."
        )

    else:

        explanation = (
            f"{main_resource} consumption is approximately "
            f"{abs(deviation):.1f}% lower than the normal "
            f"level for {building}. The AI model identified "
            f"this as an unusual usage pattern."
        )

    action_map = {
        "Electricity":
            resources["Electricity_kWh"]["action"],
        "Water":
            resources["Water_Liters"]["action"],
        "Waste":
            resources["Waste_kg"]["action"],
        "Paper":
            resources["Paper_kg"]["action"]
    }

    detailed_anomalies.append({
        "Date": row["Date"].strftime("%Y-%m-%d"),
        "Building": building,
        "Main Resource": main_resource,
        "Actual Usage": round(actual, 2),
        "Expected Usage": round(normal, 2),
        "Deviation": f"{deviation:+.1f}%",
        "Severity": severity,
        "AI Explanation": explanation,
        "Recommended Action":
            action_map[main_resource]
    })

detailed_anomaly_df = pd.DataFrame(
    detailed_anomalies
)


# =========================================================
# IBM GRANITE + OLLAMA FUNCTIONS
# =========================================================

def build_sustainability_context():

    context_lines = []

    context_lines.append(
        "CAMPUS RESOURCE SUMMARY"
    )

    for feature in features:

        analysis = resource_analysis.get(
            feature,
            {}
        )

        context_lines.append(
            f"- {resources[feature]['name']}: "
            f"recent={analysis.get('recent', 0):.2f} "
            f"{resources[feature]['unit']}, "
            f"historical={analysis.get('historical', 0):.2f} "
            f"{resources[feature]['unit']}, "
            f"change={analysis.get('change', 0):+.1f}%"
        )

    context_lines.append("")
    context_lines.append(
        "ANOMALY SUMMARY"
    )

    context_lines.append(
        f"- Total detected anomalies: {total_alerts}"
    )

    context_lines.append(
        f"- High: {high_alerts}"
    )

    context_lines.append(
        f"- Medium: {medium_alerts}"
    )

    context_lines.append(
        f"- Low: {low_alerts}"
    )

    if not detailed_anomaly_df.empty:

        context_lines.append("")
        context_lines.append(
            "IMPORTANT ANOMALIES"
        )

        for _, row in detailed_anomaly_df.head(5).iterrows():

            context_lines.append(
                f"- {row['Building']} | "
                f"{row['Main Resource']} | "
                f"{row['Deviation']} | "
                f"{row['Severity']}"
            )

    return "\n".join(
        context_lines
    )


def build_granite_prompt(user_question):

    context = build_sustainability_context()

    return f"""
You are an AI Sustainability Advisor for a college campus.

Use ONLY the campus evidence supplied below.

Do not invent measurements, costs, causes or facts.

USER QUESTION:
{user_question}

CAMPUS EVIDENCE:
{context}

Instructions:

1. Answer the user's question directly.
2. Clearly distinguish observed data from possible causes.
3. Mention relevant evidence and percentages.
4. Give 2 to 3 practical actions.
5. Do not claim actions have already happened.
6. If evidence is insufficient, say verification is required.
7. Treat the response as decision-support information.
8. Mention SDG 6, SDG 7 or SDG 12 when appropriate.

Use exactly this structure:

### AI Assessment

### Evidence

### Action Priority

### Responsible Team

### Recommended Actions

### Expected Impact

### Responsible AI Note

Keep the response concise and professional.
""".strip()


def call_ollama_granite(prompt):

    url = (
        "http://127.0.0.1:11434/api/chat"
    )

    model_name = "granite4.1:3b"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise college-campus "
                    "sustainability decision-support advisor. "
                    "Use only supplied evidence. "
                    "Never invent facts or measurements. "
                    "Give practical actions. "
                    "Keep responses professional."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 120,
            "top_p": 0.8
        }
    }

    try:

        request = urllib.request.Request(
            url,
            data=json.dumps(
                payload
            ).encode("utf-8"),
            headers={
                "Content-Type":
                    "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=300
        ) as response:

            result = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

        answer = str(
            result.get(
                "message",
                {}
            ).get(
                "content",
                ""
            )
        ).strip()

        if not answer:

            return (
                None,
                "Ollama returned an empty response."
            )

        return answer, None

    except urllib.error.HTTPError as exc:

        try:
            error_body = exc.read().decode(
                "utf-8",
                errors="replace"
            )
        except Exception:
            error_body = str(exc)

        return (
            None,
            f"Ollama HTTP {exc.code}: "
            f"{error_body[:1000]}"
        )

    except urllib.error.URLError as exc:

        return (
            None,
            "Ollama could not be reached. "
            "Make sure Ollama is running and "
            "IBM Granite 4.1 3B is installed. "
            f"Details: {exc}"
        )

    except Exception as exc:

        return (
            None,
            f"{type(exc).__name__}: "
            f"{str(exc)[:1000]}"
        )


def local_grounded_response(
    user_question
):

    normalized = str(
        user_question
    ).lower()

    detected_feature = None

    if any(
        term in normalized
        for term in [
            "electricity",
            "power",
            "energy",
            "lighting",
            "light",
            "hvac",
            "cooling",
            "computer",
            "equipment"
        ]
    ):

        detected_feature = (
            "Electricity_kWh"
        )

    elif any(
        term in normalized
        for term in [
            "water",
            "leak",
            "leakage",
            "pipe",
            "pipeline",
            "tap",
            "overflow"
        ]
    ):

        detected_feature = (
            "Water_Liters"
        )

    elif any(
        term in normalized
        for term in [
            "waste",
            "garbage",
            "trash",
            "rubbish",
            "recycle",
            "recycling"
        ]
    ):

        detected_feature = (
            "Waste_kg"
        )

    elif any(
        term in normalized
        for term in [
            "paper",
            "printing",
            "print",
            "document"
        ]
    ):

        detected_feature = (
            "Paper_kg"
        )

    if detected_feature:

        analysis = resource_analysis[
            detected_feature
        ]

        resource_name = resources[
            detected_feature
        ]["name"]

        change = analysis[
            "change"
        ]

        if abs(change) > 10:

            priority = "High"

        elif abs(change) > 5:

            priority = "Medium"

        else:

            priority = "Low"

        team_map = {
            "Electricity_kWh":
                "Maintenance Team",
            "Water_Liters":
                "Maintenance Team",
            "Waste_kg":
                "Sustainability Coordinators",
            "Paper_kg":
                "Campus Administration"
        }

        if change >= 0:

            evidence = (
                f"{resource_name} is "
                f"{change:+.1f}% above "
                f"its historical baseline."
            )

        else:

            evidence = (
                f"{resource_name} is "
                f"{abs(change):.1f}% below "
                f"its historical baseline."
            )

        return (
            "### AI Assessment\n\n"
            f"The question is primarily related to "
            f"**{resource_name}**.\n\n"

            "### Evidence\n\n"
            f"{evidence}\n\n"
            f"Detected AI anomalies: **{total_alerts}**.\n\n"

            "### Action Priority\n\n"
            f"**{priority}** — based on the current "
            "prototype decision-support thresholds.\n\n"

            "### Responsible Team\n\n"
            f"{team_map[detected_feature]}\n\n"

            "### Recommended Actions\n\n"
            f"1. {resources[detected_feature]['action']}\n"
            "2. Continue monitoring after corrective action.\n"
            "3. Verify significant deviations before "
            "operational changes.\n\n"

            "### Expected Impact\n\n"
            "Appropriate corrective action may reduce "
            "unnecessary resource consumption and "
            "support campus sustainability goals.\n\n"

            "### Responsible AI Note\n\n"
            "This is a grounded decision-support response. "
            "Important operational decisions should be "
            "verified by responsible campus personnel."
        )

    return (
        "### AI Assessment\n\n"
        "The question does not clearly identify a "
        "specific campus resource.\n\n"

        "### Evidence\n\n"
        f"The selected dataset contains "
        f"**{total_alerts}** detected anomalies.\n\n"

        "### Action Priority\n\n"
        "Low — insufficient resource-specific evidence.\n\n"

        "### Responsible Team\n\n"
        "Campus Administration / Sustainability Team.\n\n"

        "### Recommended Actions\n\n"
        "Ask about electricity, water, waste, paper, "
        "buildings, anomalies or resource reduction.\n\n"

        "### Expected Impact\n\n"
        "Resource-specific analysis can help prioritize "
        "sustainability improvements.\n\n"

        "### Responsible AI Note\n\n"
        "AI recommendations should be verified by "
        "appropriate campus personnel."
    )


# =========================================================
# KNOWLEDGE BASE SEARCH
# =========================================================

def normalize_query(text):

    text = str(text).lower()

    replacements = {
        "power": "electricity",
        "electric": "electricity",
        "energy": "electricity",
        "bill": "electricity",

        "water leak": "leakage",
        "leaking": "leakage",
        "pipe": "leakage",
        "pipeline": "leakage",
        "tap leak": "leakage",
        "overflow": "leakage",

        "garbage": "waste",
        "trash": "waste",
        "rubbish": "waste",
        "litter": "waste",

        "recycled": "recycling",
        "recyclable": "recycling",

        "cooling": "hvac",
        "air conditioning": "hvac",
        "air conditioner": "hvac",
        "ac": "hvac",

        "printing": "paper",
        "print": "paper",
        "printed": "paper",
        "documents": "paper",
        "document": "paper",
        "paperwork": "paper",

        "lights": "lighting",
        "light": "lighting",

        "computers": "equipment",
        "computer": "equipment",
        "machines": "equipment",
        "machine": "equipment",
        "devices": "equipment",
        "device": "equipment"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    return text


def search_knowledge_base(
    user_question,
    top_n=3
):

    query = normalize_query(
        user_question
    )

    if knowledge_df.empty:
        return pd.DataFrame()

    records = []

    query_words = set(
        re.findall(
            r"[a-zA-Z]+",
            query
        )
    )

    for index, row in knowledge_df.iterrows():

        searchable = " ".join([
            str(row.get("Topic", "")),
            str(row.get("Resource", "")),
            str(row.get("Guidance", "")),
            str(row.get("Recommended For", "")),
            str(row.get("Alignment", ""))
        ]).lower()

        searchable = normalize_query(
            searchable
        )

        score = 0

        for word in query_words:

            if len(word) < 3:
                continue

            if word in searchable:
                score += 1

        if (
            "electricity" in query
            and "electricity" in searchable
        ):
            score += 3

        if (
            "water" in query
            and "water" in searchable
        ):
            score += 3

        if (
            "waste" in query
            and "waste" in searchable
        ):
            score += 3

        if (
            "paper" in query
            and "paper" in searchable
        ):
            score += 3

        if (
            "leakage" in query
            and (
                "leak" in searchable
                or "water" in searchable
            )
        ):
            score += 3

        if (
            "recycling" in query
            and "recycl" in searchable
        ):
            score += 3

        if (
            "hvac" in query
            and "hvac" in searchable
        ):
            score += 3

        if score > 0:

            records.append({
                "index": index,
                "score": score
            })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(
        records
    ).sort_values(
        "score",
        ascending=False
    ).head(top_n)

    return knowledge_df.loc[
        result["index"]
    ]


# =========================================================
# IMPACT ASSUMPTIONS
# =========================================================

IMPACT_ASSUMPTIONS = {
    "Electricity_kWh": {
        "cost_per_unit": 9.0,
        "co2_per_unit": 0.70,
        "cost_unit": "₹/kWh",
        "co2_unit": "kg CO₂e/kWh"
    },
    "Water_Liters": {
        "cost_per_unit": 0.05,
        "co2_per_unit": 0.0003,
        "cost_unit": "₹/L",
        "co2_unit": "kg CO₂e/L"
    },
    "Waste_kg": {
        "cost_per_unit": 4.0,
        "co2_per_unit": 0.50,
        "cost_unit": "₹/kg",
        "co2_unit": "kg CO₂e/kg"
    },
    "Paper_kg": {
        "cost_per_unit": 60.0,
        "co2_per_unit": 1.30,
        "cost_unit": "₹/kg",
        "co2_unit": "kg CO₂e/kg"
    }
}


# =========================================================
# TABS
# =========================================================

(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7,
    tab8,
    tab9,
    tab10,
    tab11
) = st.tabs(
    [
        "📊 Overview",
        "🚨 AI Anomaly Detection",
        "🤖 AI Sustainability Advisor",
        "💡 Recommendations",
        "🎯 Impact Simulator",
        "📋 Data",
        "🌍 Performance Score",
        "🎯 Action Center",
        "🌱 Goal Tracker",
        "🔮 Forecast",
        "📄 Sustainability Report"
    ]
)


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tab1:

    st.header(
        "📊 Campus Sustainability Overview"
    )

    st.caption(
        "Monitor campus resource consumption, AI findings "
        "and sustainability performance."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🌱 Sustainability Score",
            f"{sustainability_score}/100"
        )

    with col2:

        st.metric(
            "🚨 AI Alerts",
            total_alerts
        )

    with col3:

        st.metric(
            "🏢 Buildings",
            view["Building"].nunique()
        )

    with col4:

        st.metric(
            "📊 Records",
            f"{len(view):,}"
        )

    st.divider()

    st.subheader(
        "📈 Resource Consumption Trends"
    )

    trend_feature = st.selectbox(
        "Select Resource",
        features,
        format_func=lambda x:
            f"{resources[x]['icon']} "
            f"{resources[x]['name']}",
        key="overview_resource"
    )

    trend_df = (
        view.groupby("Date")[
            trend_feature
        ]
        .sum()
        .reset_index()
    )

    fig = px.line(
        trend_df,
        x="Date",
        y=trend_feature,
        markers=True,
        title=(
            f"{resources[trend_feature]['name']} "
            "Consumption Trend"
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=(
            f"{resources[trend_feature]['name']} "
            f"({resources[trend_feature]['unit']})"
        ),
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "📊 Resource Status"
    )

    status_rows = []

    for feature in features:

        change = resource_analysis[
            feature
        ]["change"]

        if change > 10:
            status = "🔴 Needs Attention"

        elif change > 5:
            status = "🟠 Monitor"

        else:
            status = "🟢 Stable"

        status_rows.append({
            "Resource":
                resources[feature]["name"],
            "Recent Change":
                f"{change:+.1f}%",
            "Status":
                status
        })

    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True
    )

    if highest_change > 0:

        st.warning(
            f"⚠️ **AI Attention:** "
            f"{highest_name} shows the highest recent "
            f"increase at {highest_change:+.1f}% "
            f"compared with its historical baseline."
        )

    else:

        st.success(
            "✅ Recent resource consumption is currently "
            "within or below the historical baseline."
        )


# =========================================================
# TAB 2 — AI ANOMALY DETECTION
# =========================================================

with tab2:

    st.header(
        "🚨 AI Anomaly Detection"
    )

    st.caption(
        "Isolation Forest identifies unusual multivariate "
        "campus resource-consumption patterns."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🚨 Total AI Alerts",
            total_alerts
        )

    with col2:

        st.metric(
            "🔴 High Severity",
            high_alerts
        )

    with col3:

        st.metric(
            "🟠 Medium Severity",
            medium_alerts
        )

    with col4:

        st.metric(
            "🟡 Low Severity",
            low_alerts
        )

    st.divider()

    st.subheader(
        "🔎 Detailed AI Findings"
    )

    if detailed_anomaly_df.empty:

        st.success(
            "No significant anomalies detected "
            "for the selected filters."
        )

    else:

        st.dataframe(
            detailed_anomaly_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "🧠 AI Investigation"
        )

        selected_index = st.selectbox(
            "Select an anomaly to investigate",
            range(
                len(detailed_anomaly_df)
            ),
            format_func=lambda x:
                (
                    f"{detailed_anomaly_df.iloc[x]['Date']} — "
                    f"{detailed_anomaly_df.iloc[x]['Building']} — "
                    f"{detailed_anomaly_df.iloc[x]['Main Resource']}"
                ),
            key="anomaly_selector"
        )

        selected = detailed_anomaly_df.iloc[
            selected_index
        ]

        st.info(
            f"**AI Finding:** "
            f"{selected['AI Explanation']}"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Actual Usage",
                selected["Actual Usage"]
            )

        with col2:

            st.metric(
                "Expected Usage",
                selected["Expected Usage"]
            )

        with col3:

            st.metric(
                "Deviation",
                selected["Deviation"]
            )

        st.markdown(
            f"**Severity:** {selected['Severity']}"
        )

        st.markdown(
            f"**Recommended Action:** "
            f"{selected['Recommended Action']}"
        )


# =========================================================
# TAB 3 — AI SUSTAINABILITY ADVISOR
# =========================================================

with tab3:

    st.header(
        "🤖 AI Sustainability Advisor"
    )

    st.caption(
        "Combine campus data, knowledge-base guidance "
        "and local IBM Granite intelligence."
    )

    # -----------------------------------------------------
    # DATA-DRIVEN ADVISOR
    # -----------------------------------------------------

    st.subheader(
        "💬 Data-Driven Advisor"
    )

    quick_questions = [
        "Which resource needs the most attention?",
        "Which building needs the most attention?",
        "Why is electricity consumption high?",
        "Why is water consumption high?",
        "Why is waste generation high?",
        "Why is paper consumption high?",
        "What should the maintenance team do?",
        "How can we reduce campus resource consumption?",
        "Give me an overall sustainability summary."
    ]

    question = st.selectbox(
        "Choose a question",
        quick_questions,
        key="quick_advisor_question"
    )

    st.divider()

    # -----------------------------------------------------
    # QUICK QUESTION RESPONSES
    # -----------------------------------------------------

    if question == (
        "Which resource needs the most attention?"
    ):

        st.success(
            f"### 🎯 AI Finding\n\n"
            f"**{highest_name}** currently requires "
            f"the most attention based on recent "
            f"change compared with the historical baseline."
        )

        st.write(
            f"Recent change: "
            f"**{highest_change:+.1f}%**"
        )

        st.info(
            f"💡 **Recommended Action:** "
            f"{resources[highest_resource]['action']}"
        )

    elif question == (
        "Which building needs the most attention?"
    ):

        if alerts.empty:

            st.success(
                "No anomalies were detected."
            )

        else:

            building_counts = (
                alerts["Building"]
                .value_counts()
            )

            building_name = (
                building_counts.index[0]
            )

            building_alerts = (
                building_counts.iloc[0]
            )

            st.warning(
                f"🏢 **{building_name}** has the "
                f"highest number of detected anomalies "
                f"with **{building_alerts}** alert(s)."
            )

            st.info(
                "The result is based on the number of "
                "AI-detected unusual records in the "
                "selected dataset."
            )

    elif question == (
        "Why is electricity consumption high?"
    ):

        change = resource_analysis[
            "Electricity_kWh"
        ]["change"]

        if change > 0:

            st.warning(
                f"⚡ Electricity consumption is "
                f"**{change:+.1f}% above** "
                "the historical baseline."
            )

        else:

            st.success(
                f"⚡ Electricity consumption is "
                f"{abs(change):.1f}% below "
                "the historical baseline."
            )

        st.info(
            "💡 Recommended Action: Check HVAC operation, "
            "lighting schedules and high-load laboratory "
            "or computing equipment."
        )

    elif question == (
        "Why is water consumption high?"
    ):

        change = resource_analysis[
            "Water_Liters"
        ]["change"]

        if change > 0:

            st.warning(
                f"💧 Water consumption is "
                f"**{change:+.1f}% above** "
                "the historical baseline."
            )

        else:

            st.success(
                f"💧 Water consumption is "
                f"{abs(change):.1f}% below "
                "the historical baseline."
            )

        st.info(
            "💡 Recommended Action: Inspect pipelines, "
            "taps, storage tanks and possible leakage "
            "or overflow points."
        )

    elif question == (
        "Why is waste generation high?"
    ):

        change = resource_analysis[
            "Waste_kg"
        ]["change"]

        st.write(
            f"♻️ Waste recent change: "
            f"**{change:+.1f}%**."
        )

        st.info(
            "💡 Recommended Action: Review major waste "
            "sources, improve segregation and strengthen "
            "recycling practices."
        )

    elif question == (
        "Why is paper consumption high?"
    ):

        change = resource_analysis[
            "Paper_kg"
        ]["change"]

        st.write(
            f"📄 Paper recent change: "
            f"**{change:+.1f}%**."
        )

        st.info(
            "💡 Recommended Action: Increase digital "
            "documentation, online forms and double-sided "
            "printing."
        )

    elif question == (
        "What should the maintenance team do?"
    ):

        st.info(
            "🔧 Maintenance Team Priorities"
        )

        st.markdown(
            """
            1. Review HVAC operating schedules.
            2. Inspect lighting in low-occupancy areas.
            3. Check high-load laboratory and computing equipment.
            4. Inspect water pipelines, taps and storage systems.
            5. Investigate significant AI-detected deviations.
            """
        )

    elif question == (
        "How can we reduce campus resource consumption?"
    ):

        st.success(
            "🌱 Recommended Sustainability Strategy"
        )

        st.markdown(
            """
            **Electricity:** Optimize HVAC and lighting schedules.

            **Water:** Detect and repair leaks and avoid unnecessary use.

            **Waste:** Improve segregation and recycling.

            **Paper:** Increase digital documentation and reduce unnecessary printing.
            """
        )

    else:

        st.success(
            "### 🌍 Overall Sustainability Summary"
        )

        for feature in features:

            analysis = resource_analysis[
                feature
            ]

            st.write(
                f"**{resources[feature]['name']}:** "
                f"{analysis['change']:+.1f}% compared "
                "with historical baseline."
            )

        st.info(
            f"Total AI-detected anomalies: "
            f"**{total_alerts}**."
        )

    # -----------------------------------------------------
    # KNOWLEDGE BASE
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "📚 Sustainability Knowledge Advisor"
    )

    knowledge_question = st.text_input(
        "Ask a sustainability knowledge question",
        placeholder=(
            "Example: How can we reduce water usage?"
        ),
        key="knowledge_question"
    )

    if knowledge_question.strip():

        matches = search_knowledge_base(
            knowledge_question,
            top_n=3
        )

        if matches.empty:

            st.warning(
                "No strong knowledge-base match was found."
            )

        else:

            st.success(
                f"Found {len(matches)} relevant "
                "knowledge-base recommendation(s)."
            )

            for _, record in matches.iterrows():

                st.markdown(
                    f"### {record['Topic']}"
                )

                st.write(
                    f"**Resource:** "
                    f"{record['Resource']}"
                )

                st.write(
                    f"**Guidance:** "
                    f"{record['Guidance']}"
                )

                st.caption(
                    f"Recommended for: "
                    f"{record['Recommended For']} | "
                    f"Alignment: "
                    f"{record['Alignment']}"
                )

                st.divider()

    # -----------------------------------------------------
    # CONVERSATIONAL GRANITE
    # -----------------------------------------------------

    st.subheader(
        "💬 Conversational AI Sustainability Advisor"
    )

    st.caption(
        "Ask a natural-language question. IBM Granite 4.1 3B "
        "generates the response locally through Ollama."
    )

    granite_question = st.text_area(
        "Ask the AI Sustainability Advisor",
        placeholder=(
            "Example: Our electricity consumption has increased. "
            "What should the maintenance team do?"
        ),
        height=110,
        key="granite_sustainability_question"
    )

    granite_col1, granite_col2 = st.columns(
        [1, 1]
    )

    with granite_col1:

        generate_response = st.button(
            "🤖 Generate AI Response",
            type="primary",
            use_container_width=True,
            key="generate_granite_response"
        )

    with granite_col2:

        with st.expander(
            "⚙️ AI Model Information"
        ):

            st.write(
                "**AI Model:** IBM Granite 4.1 3B"
            )

            st.write(
                "**Runtime:** Ollama"
            )

            st.write(
                "**Inference:** Local"
            )

            st.write(
                "**Local API:** "
                "`http://127.0.0.1:11434`"
            )

            st.success(
                "IBM Granite 4.1 3B is configured "
                "for local inference."
            )

            st.caption(
                "No IBM Cloud API key or "
                "watsonx.ai credentials are required."
            )

    if generate_response:

        if not granite_question.strip():

            st.warning(
                "Please enter a sustainability "
                "question first."
            )

        else:

            with st.spinner(
                "IBM Granite is analyzing the "
                "campus sustainability context..."
            ):

                try:

                    prompt = build_granite_prompt(
                        granite_question.strip()
                    )

                    granite_response, granite_error = (
                        call_ollama_granite(prompt)
                    )

                except Exception as exc:

                    granite_response = None

                    granite_error = (
                        f"{type(exc).__name__}: "
                        f"{str(exc)}"
                    )

            if granite_response:

                st.success(
                    "✅ IBM Granite response "
                    "generated successfully."
                )

                st.markdown(
                    granite_response
                )

                st.caption(
                    "Model: IBM Granite 4.1 3B via Ollama | "
                    "Grounding: campus dataset + "
                    "sustainability knowledge base"
                )

            else:

                st.error(
                    "❌ IBM Granite could not generate "
                    "a response."
                )

                st.markdown(
                    local_grounded_response(
                        granite_question.strip()
                    )
                )

                if granite_error:

                    with st.expander(
                        "🔎 Ollama / Granite error details",
                        expanded=True
                    ):

                        st.code(
                            granite_error,
                            language="text"
                        )

            st.warning(
                "Responsible AI: This assistant provides "
                "decision-support information. Verify "
                "significant findings and operational "
                "recommendations with appropriate campus "
                "personnel."
            )


# =========================================================
# TAB 4 — RECOMMENDATIONS
# =========================================================

with tab4:

    st.header(
        "💡 AI Sustainability Recommendations"
    )

    st.caption(
        "Recommendations are based on recent consumption "
        "patterns and AI-detected deviations."
    )

    recommendation_rows = []

    for feature in features:

        change = resource_analysis[
            feature
        ]["change"]

        if change > 10:
            priority = "🔴 High"

        elif change > 5:
            priority = "🟡 Medium"

        else:
            priority = "🟢 Low"

        recommendation_rows.append({
            "Resource":
                resources[feature]["name"],
            "Recent Change":
                f"{change:+.1f}%",
            "Priority":
                priority,
            "Recommended Action":
                resources[feature]["action"]
        })

    recommendation_df = pd.DataFrame(
        recommendation_rows
    )

    st.dataframe(
        recommendation_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "🎯 Priority Recommendation"
    )

    st.info(
        f"**{resources[highest_resource]['icon']} "
        f"{highest_name}** currently shows the "
        f"highest recent change of "
        f"**{highest_change:+.1f}%**."
    )

    st.write(
        resources[highest_resource]["action"]
    )

    st.warning(
        "Recommendations are decision-support information "
        "and should be verified before operational action."
    )


# =========================================================
# TAB 5 — IMPACT SIMULATOR
# =========================================================

with tab5:

    st.header(
        "🎯 Sustainability Impact Simulator"
    )

    st.caption(
        "Estimate potential resource, cost and environmental "
        "benefits from reducing campus consumption."
    )

    st.info(
        "The simulator uses illustrative prototype assumptions. "
        "Replace these with verified campus-specific values "
        "before real-world use."
    )

    reduction_cols = st.columns(4)

    reduction_values = {}

    for i, feature in enumerate(features):

        with reduction_cols[i]:

            reduction_values[feature] = st.slider(
                resources[feature]["name"],
                min_value=0,
                max_value=30,
                value=10,
                step=5,
                format="%d%%",
                key=f"impact_{feature}"
            )

    impact_rows = []

    for feature in features:

        current_usage = float(
            view[feature].sum()
        )

        reduction_percent = (
            reduction_values[feature]
        )

        saved_usage = (
            current_usage *
            reduction_percent /
            100
        )

        remaining_usage = (
            current_usage -
            saved_usage
        )

        assumptions = IMPACT_ASSUMPTIONS[
            feature
        ]

        cost_saving = (
            saved_usage *
            assumptions["cost_per_unit"]
        )

        co2_reduction = (
            saved_usage *
            assumptions["co2_per_unit"]
        )

        impact_rows.append({
            "Resource":
                resources[feature]["name"],
            "Current Usage":
                current_usage,
            "Reduction":
                f"{reduction_percent}%",
            "Potential Saving":
                saved_usage,
            "Remaining Usage":
                remaining_usage,
            "Cost Saving":
                cost_saving,
            "CO₂ Reduction":
                co2_reduction
        })

    impact_df = pd.DataFrame(
        impact_rows
    )

    total_cost_saving = impact_df[
        "Cost Saving"
    ].sum()

    total_co2_reduction = impact_df[
        "CO₂ Reduction"
    ].sum()

    total_resource_saving = impact_df[
        "Potential Saving"
    ].sum()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "💰 Potential Cost Saving",
            f"₹{total_cost_saving:,.2f}"
        )

    with col2:

        st.metric(
            "🌱 Potential CO₂ Reduction",
            f"{total_co2_reduction:,.2f} kg"
        )

    with col3:

        st.metric(
            "📉 Total Resource Saving",
            f"{total_resource_saving:,.2f}"
        )

    st.subheader(
        "📊 Impact Summary"
    )

    st.dataframe(
        impact_df,
        use_container_width=True,
        hide_index=True
    )

    chart_df = impact_df[
        [
            "Resource",
            "Current Usage",
            "Remaining Usage"
        ]
    ].copy()

    chart_long = chart_df.melt(
        id_vars=["Resource"],
        var_name="Usage Type",
        value_name="Usage"
    )

    fig_impact = px.bar(
        chart_long,
        x="Resource",
        y="Usage",
        color="Usage Type",
        barmode="group",
        title="Current vs Remaining Resource Usage"
    )

    st.plotly_chart(
        fig_impact,
        use_container_width=True
    )

    with st.expander(
        "📌 View Simulator Assumptions"
    ):

        for feature in features:

            assumption = IMPACT_ASSUMPTIONS[
                feature
            ]

            st.write(
                f"**{resources[feature]['name']}** — "
                f"{assumption['cost_unit']}, "
                f"{assumption['co2_unit']}"
            )

        st.caption(
            "These values are illustrative prototype assumptions."
        )


# =========================================================
# TAB 6 — DATA
# =========================================================

with tab6:

    st.header(
        "📋 Campus Resource Dataset"
    )

    st.caption(
        "Filtered data used by the dashboard and AI analysis."
    )

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=view.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="filtered_campus_resource_data.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.info(
        "The current dataset is simulated for development "
        "and testing. Future versions can connect to "
        "verified campus utility records."
    )


# =========================================================
# TAB 7 — PERFORMANCE SCORE
# =========================================================

with tab7:

    st.header(
        "🌍 AI Sustainability Performance"
    )

    st.caption(
        "A transparent prototype indicator combining "
        "recent resource-consumption changes and "
        "AI-detected unusual patterns."
    )

    performance_rows = []

    for feature in features:

        recent_avg = resource_analysis[
            feature
        ]["recent"]

        historical_avg = resource_analysis[
            feature
        ]["historical"]

        change_pct = resource_analysis[
            feature
        ]["change"]

        single_model = IsolationForest(
            contamination=0.03,
            random_state=42
        )

        labels = single_model.fit_predict(
            view[[feature]]
        )

        anomaly_rate = float(
            np.mean(
                labels == -1
            ) * 100
        )

        increase_penalty = min(
            60.0,
            max(change_pct, 0.0) * 2.0
        )

        anomaly_penalty = min(
            40.0,
            anomaly_rate * 4.0
        )

        score = int(
            np.clip(
                100 -
                increase_penalty -
                anomaly_penalty,
                0,
                100
            )
        )

        if score >= 80:

            status = "🟢 Strong"

        elif score >= 60:

            status = "🟡 Monitor"

        else:

            status = "🔴 Needs Attention"

        performance_rows.append({
            "Resource":
                resources[feature]["name"],
            "Score":
                score,
            "Recent Change":
                f"{change_pct:+.1f}%",
            "Anomaly Rate":
                f"{anomaly_rate:.1f}%",
            "Status":
                status
        })

    performance_df = pd.DataFrame(
        performance_rows
    )

    overall_score = int(
        round(
            performance_df["Score"].mean()
        )
    )

    strongest_resource = (
        performance_df.loc[
            performance_df["Score"].idxmax(),
            "Resource"
        ]
    )

    priority_resource = (
        performance_df.loc[
            performance_df["Score"].idxmin(),
            "Resource"
        ]
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🌍 Overall Performance",
            f"{overall_score}/100"
        )

    with col2:

        st.metric(
            "⭐ Strongest Resource",
            strongest_resource
        )

    with col3:

        st.metric(
            "⚠️ Priority Resource",
            priority_resource
        )

    st.progress(
        max(
            0,
            min(
                100,
                overall_score
            )
        ) / 100
    )

    if overall_score >= 80:

        st.success(
            "Overall performance is currently strong "
            "according to the prototype indicator."
        )

    elif overall_score >= 60:

        st.warning(
            "Overall performance should be monitored "
            "for improvement opportunities."
        )

    else:

        st.error(
            "Overall performance requires attention "
            "according to the prototype indicator."
        )

    st.subheader(
        "📊 Resource Performance Scores"
    )

    score_chart = px.bar(
        performance_df,
        x="Resource",
        y="Score",
        text="Score",
        range_y=[0, 100],
        title="AI Sustainability Performance by Resource"
    )

    score_chart.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        score_chart,
        use_container_width=True
    )

    st.subheader(
        "📋 Performance Breakdown"
    )

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    weakest_row = performance_df.loc[
        performance_df["Score"].idxmin()
    ]

    st.subheader(
        "🤖 AI Performance Interpretation"
    )

    st.write(
        f"The resource requiring the greatest attention "
        f"is **{weakest_row['Resource']}**, with a "
        f"performance score of "
        f"**{weakest_row['Score']}/100**."
    )

    recommended_focus = next(
    (
        resources[f]["action"]
        for f in features
        if resources[f]["name"] == weakest_row["Resource"]
    ),
    "Review the resource consumption pattern and investigate the main contributing factors."
)

st.info(
    f"💡 Recommended Focus: {recommended_focus}"
)

st.caption(
        "Scoring method: each resource starts at 100. "
        "Positive recent consumption change can reduce "
        "the score by up to 60 points, while detected "
        "unusual-pattern rate can reduce it by up to "
        "40 points. The overall score is the average "
        "of the four resource scores."
    )

st.warning(
        "Responsible AI: This score is a prototype "
        "decision-support indicator, not an official "
        "environmental rating."
    )


# =========================================================
# TAB 8 — ACTION CENTER
# =========================================================

with tab8:

    st.header(
        "🎯 AI Sustainability Action Center"
    )

    st.caption(
        "Convert AI findings into prioritized actions, "
        "responsible teams and implementation tracking."
    )

    team_map = {
        "Electricity":
            "Maintenance Team",
        "Water":
            "Maintenance Team",
        "Waste":
            "Sustainability Coordinators",
        "Paper":
            "Campus Administration"
    }

    action_map = {
        "Electricity":
            "Check HVAC schedules, lighting in unoccupied "
            "areas and high-load equipment.",
        "Water":
            "Inspect pipelines, taps, storage tanks and "
            "possible leakage or overflow points.",
        "Waste":
            "Review major waste sources, improve segregation "
            "and strengthen recycling practices.",
        "Paper":
            "Increase digital documentation, online forms "
            "and double-sided printing."
    }

    action_rows = []

    for _, row in performance_df.iterrows():

        resource_name = row[
            "Resource"
        ]

        score = float(
            row["Score"]
        )

        change = float(
            str(
                row["Recent Change"]
            ).replace(
                "%",
                ""
            )
        )

        anomaly_rate = float(
            str(
                row["Anomaly Rate"]
            ).replace(
                "%",
                ""
            )
        )

        if (
            score < 60
            or change > 10
            or anomaly_rate > 10
        ):

            priority = "🔴 High"

        elif (
            score < 80
            or change > 5
            or anomaly_rate > 5
        ):

            priority = "🟡 Medium"

        else:

            priority = "🟢 Low"

        action_rows.append({
            "Resource":
                resource_name,
            "Priority":
                priority,
            "Performance Score":
                score,
            "Recent Change":
                f"{change:+.1f}%",
            "Anomaly Rate":
                f"{anomaly_rate:.1f}%",
            "Responsible Team":
                team_map.get(
                    resource_name,
                    "Campus Administration"
                ),
            "Recommended Action":
                action_map.get(
                    resource_name,
                    "Review resource usage."
                )
        })

    action_df = pd.DataFrame(
        action_rows
    )

    high_count = int(
        action_df["Priority"]
        .str.contains("High")
        .sum()
    )

    medium_count = int(
        action_df["Priority"]
        .str.contains("Medium")
        .sum()
    )

    low_count = int(
        action_df["Priority"]
        .str.contains("Low")
        .sum()
    )

    total_actions = len(
        action_df
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🔴 High Priority",
            high_count
        )

    with col2:

        st.metric(
            "🟡 Medium Priority",
            medium_count
        )

    with col3:

        st.metric(
            "🟢 Low Priority",
            low_count
        )

    with col4:

        st.metric(
            "📌 Total Actions",
            total_actions
        )

    st.divider()

    st.subheader(
        "📋 Prioritized Action Plan"
    )

    st.dataframe(
        action_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "📌 Recommended Immediate Focus"
    )

    priority_order = {
        "🔴 High": 1,
        "🟡 Medium": 2,
        "🟢 Low": 3
    }

    action_df["_rank"] = (
        action_df["Priority"]
        .map(priority_order)
        .fillna(3)
    )

    sorted_actions = (
        action_df
        .sort_values(
            ["_rank", "Performance Score"]
        )
    )

    first_action = sorted_actions.iloc[0]

    st.info(
        f"**{first_action['Resource']} — "
        f"{first_action['Priority']}**\n\n"
        f"Responsible Team: "
        f"**{first_action['Responsible Team']}**\n\n"
        f"{first_action['Recommended Action']}"
    )

    st.download_button(
        "⬇️ Download Action Plan CSV",
        data=action_df.drop(
            columns=["_rank"],
            errors="ignore"
        ).to_csv(
            index=False
        ).encode("utf-8"),
        file_name="campus_sustainability_action_plan.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.warning(
        "Responsible AI: Action priorities are prototype "
        "decision-support outputs. Campus personnel should "
        "validate actions before implementation."
    )


# =========================================================
# TAB 9 — GOAL TRACKER
# =========================================================

with tab9:

    st.header(
        "🌱 Sustainability Goal Tracker"
    )

    st.caption(
        "Track progress toward user-defined resource-reduction targets."
    )

    st.info(
        "Targets are configurable prototype goals. "
        "Replace them with officially approved campus "
        "sustainability targets when available."
    )

    # -----------------------------------------------------
    # DEFAULT GOALS
    # -----------------------------------------------------

    goal_defaults = {
        "Electricity": 10,
        "Water": 10,
        "Waste": 10,
        "Paper": 10
    }

    # Map display resource names to dataset feature names
    goal_feature_map = {
        "Electricity": "Electricity_kWh",
        "Water": "Water_Liters",
        "Waste": "Waste_kg",
        "Paper": "Paper_kg"
    }

    # -----------------------------------------------------
    # SET REDUCTION TARGETS
    # -----------------------------------------------------

    st.subheader(
        "🎯 Set Reduction Targets"
    )

    goal_cols = st.columns(4)

    goal_targets = {}

    for i, resource_name in enumerate(
        goal_defaults.keys()
    ):

        with goal_cols[i]:

            goal_targets[
                resource_name
            ] = st.slider(
                f"{resource_name} Target",
                min_value=0,
                max_value=50,
                value=goal_defaults[
                    resource_name
                ],
                step=5,
                format="%d%%",
                key=f"goal_{resource_name}"
            )

    # -----------------------------------------------------
    # CALCULATE GOAL PERFORMANCE
    # -----------------------------------------------------

    goal_rows = []

    for resource_name in goal_defaults:

        feature = goal_feature_map[
            resource_name
        ]

        analysis = resource_analysis[
            feature
        ]

        # Recent and historical values
        recent = float(
            analysis["recent"]
        )

        historical = float(
            analysis["historical"]
        )

        # User-selected target
        target_reduction = float(
            goal_targets[
                resource_name
            ]
        )

        # -------------------------------------------------
        # TARGET USAGE
        # -------------------------------------------------

        if historical != 0:

            target_usage = (
                historical *
                (
                    1 -
                    target_reduction / 100
                )
            )

        else:

            target_usage = 0.0

        # -------------------------------------------------
        # ACHIEVED REDUCTION
        # -------------------------------------------------

        if historical != 0:

            achieved_reduction = (
                (
                    historical - recent
                )
                / historical
            ) * 100

        else:

            achieved_reduction = 0.0

        # -------------------------------------------------
        # REQUIRED IMPROVEMENT
        #
        # Example:
        # Target = 10%
        # Achieved = -1.8%
        #
        # Required improvement =
        # 10 - (-1.8) = 11.8%
        # -------------------------------------------------

        if target_reduction > 0:

            required_improvement = (
                target_reduction -
                achieved_reduction
            )

        else:

            required_improvement = 0.0

        required_improvement = max(
            required_improvement,
            0.0
        )

        # -------------------------------------------------
        # GOAL PROGRESS
        # -------------------------------------------------

        if target_reduction > 0:

            progress = (
                achieved_reduction /
                target_reduction
            ) * 100

        else:

            progress = 0.0

        progress = float(
            np.clip(
                progress,
                0,
                100
            )
        )

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        if target_reduction == 0:

            status = "🟢 No Target"

        elif achieved_reduction >= target_reduction:

            status = "🟢 On Track"

        elif achieved_reduction > 0:

            status = "🟡 Needs Attention"

        else:

            status = "🔴 Off Track"

        # -------------------------------------------------
        # STORE RESULT
        # -------------------------------------------------

        goal_rows.append({

            "Resource":
                resource_name,

            "Historical Baseline":
                round(
                    historical,
                    2
                ),

            "Current Usage":
                round(
                    recent,
                    2
                ),

            "Target Reduction":
                f"{target_reduction:.0f}%",

            "Target Usage":
                round(
                    target_usage,
                    2
                ),

            "Achieved Reduction":
                f"{achieved_reduction:+.1f}%",

            "Required Improvement":
                f"{required_improvement:.1f}%",

            "Progress":
                f"{progress:.1f}%",

            "Status":
                status
        })

    # -----------------------------------------------------
    # GOAL DATAFRAME
    # -----------------------------------------------------

    goal_df = pd.DataFrame(
        goal_rows
    )

    # -----------------------------------------------------
    # GOAL PROGRESS TABLE
    # -----------------------------------------------------

    st.subheader(
        "📊 Goal Progress"
    )

    st.dataframe(
        goal_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # OVERALL GOAL PROGRESS
    # -----------------------------------------------------

    progress_values = []

    for _, row in goal_df.iterrows():

        progress_value = float(
            str(
                row["Progress"]
            ).replace(
                "%",
                ""
            )
        )

        progress_values.append(
            progress_value
        )

    if progress_values:

        overall_goal_progress = float(
            np.mean(
                progress_values
            )
        )

    else:

        overall_goal_progress = 0.0

    # -----------------------------------------------------
    # OVERALL PROGRESS
    # -----------------------------------------------------

    st.subheader(
        "🌱 Overall Goal Progress"
    )

    st.metric(
        "Overall Goal Progress",
        f"{overall_goal_progress:.1f}%"
    )

    st.progress(
        float(
            np.clip(
                overall_goal_progress,
                0,
                100
            )
        ) / 100
    )

    # -----------------------------------------------------
    # GOAL STATUS MESSAGE
    # -----------------------------------------------------

    if overall_goal_progress >= 75:

        st.success(
            "🌟 The campus is making strong progress "
            "toward the configured reduction goals."
        )

    elif overall_goal_progress >= 40:

        st.info(
            "🌱 Moderate progress is visible. "
            "Continue monitoring resource performance."
        )

    else:

        st.warning(
            "⚠️ Goal progress is currently limited. "
            "Current consumption is above the desired "
            "reduction targets for some resources. "
            "Progress will increase when actual consumption "
            "moves toward the configured targets."
        )

    # -----------------------------------------------------
    # TARGET INTERPRETATION
    # -----------------------------------------------------

    st.subheader(
        "🔎 Goal Interpretation"
    )

    # Find resource requiring the largest improvement
    interpretation_rows = []

    for _, row in goal_df.iterrows():

        required_value = float(
            str(
                row["Required Improvement"]
            ).replace(
                "%",
                ""
            )
        )

        interpretation_rows.append(
            (
                row["Resource"],
                required_value
            )
        )

    if interpretation_rows:

        priority_resource, priority_required = max(
            interpretation_rows,
            key=lambda x: x[1]
        )

        priority_row = goal_df[
            goal_df["Resource"] ==
            priority_resource
        ].iloc[0]

        st.info(
            f"💡 **Priority Resource: {priority_resource}**\n\n"
            f"Current usage: "
            f"{priority_row['Current Usage']}\n\n"
            f"Configured target: "
            f"{priority_row['Target Reduction']}\n\n"
            f"Required improvement: "
            f"{priority_required:.1f}%\n\n"
            f"Status: "
            f"{priority_row['Status']}"
        )

    # -----------------------------------------------------
    # DOWNLOAD GOAL PROGRESS
    # -----------------------------------------------------

    st.download_button(
        "⬇️ Download Goal Progress CSV",
        data=goal_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "campus_sustainability_goal_progress.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )

    # -----------------------------------------------------
    # RESPONSIBLE AI
    # -----------------------------------------------------

    st.warning(
        "Responsible AI: Goal progress is a prototype "
        "decision-support indicator based on the selected "
        "targets and available campus resource data. "
        "Targets should be validated and approved by "
        "appropriate campus personnel."
    )

# =========================================================
# TAB 10 — SUSTAINABILITY FORECAST
# =========================================================

with tab10:

    st.header(
        "🔮 Sustainability Forecast"
    )

    st.caption(
        "Machine-learning predictions estimate potential "
        "resource-consumption patterns for the next seven days."
    )

    st.info(
        "Forecast values are predictive estimates based on "
        "historical dataset patterns and should be validated "
        "against actual measurements."
    )

    forecast_resources = {
        "Electricity": {
            "feature": "Electricity_kWh",
            "unit": "kWh"
        },
        "Water": {
            "feature": "Water_Liters",
            "unit": "Liters"
        },
        "Waste": {
            "feature": "Waste_kg",
            "unit": "kg"
        },
        "Paper": {
            "feature": "Paper_kg",
            "unit": "kg"
        }
    }

    forecast_days = st.slider(
        "Historical days used for forecasting",
        min_value=7,
        max_value=60,
        value=30,
        step=1,
        key="forecast_history_days"
    )

    daily_data = (
        view.groupby("Date")[
            [
                "Electricity_kWh",
                "Water_Liters",
                "Waste_kg",
                "Paper_kg"
            ]
        ]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    if len(daily_data) < 7:

        st.warning(
            "Not enough historical data is available "
            "to generate a forecast."
        )

    else:

        forecast_history = (
            daily_data.tail(
                min(
                    forecast_days,
                    len(daily_data)
                )
            ).copy()
        )

        def generate_forecast(
            data,
            feature
        ):

            values = data[
                feature
            ].astype(float).values

            x = np.arange(
                len(values)
            )

            model = LinearRegression()

            model.fit(
                x.reshape(-1, 1),
                values
            )

            future_x = np.arange(
                len(values),
                len(values) + 7
            )

            predictions = model.predict(
                future_x.reshape(-1, 1)
            )

            predictions = np.maximum(
                predictions,
                0
            )

            split = max(
                1,
                len(values) // 3
            )

            first_avg = np.mean(
                values[:split]
            )

            last_avg = np.mean(
                values[-split:]
            )

            if first_avg != 0:

                trend_change = (
                    (
                        last_avg -
                        first_avg
                    )
                    / first_avg
                ) * 100

            else:

                trend_change = 0.0

            if trend_change > 3:

                trend = "📈 Increasing"

            elif trend_change < -3:

                trend = "📉 Decreasing"

            else:

                trend = "➡️ Stable"

            return (
                predictions,
                trend,
                trend_change
            )

        forecast_results = {}

        for resource_name, config in (
            forecast_resources.items()
        ):

            predictions, trend, trend_change = (
                generate_forecast(
                    forecast_history,
                    config["feature"]
                )
            )

            forecast_results[
                resource_name
            ] = {
                "predictions":
                    predictions,
                "trend":
                    trend,
                "trend_change":
                    trend_change,
                "unit":
                    config["unit"]
            }

        st.subheader(
            "📊 Forecast Summary"
        )

        summary_cols = st.columns(4)

        for i, resource_name in enumerate(
            forecast_resources.keys()
        ):

            result = forecast_results[
                resource_name
            ]

            with summary_cols[i]:

                st.metric(
                    resource_name,
                    result["trend"],
                    f"{result['trend_change']:+.1f}% trend"
                )

        st.subheader(
            "📈 Resource Forecast"
        )

        selected_resource = st.selectbox(
            "Select a resource",
            list(
                forecast_resources.keys()
            ),
            key="forecast_resource"
        )

        selected_result = (
            forecast_results[
                selected_resource
            ]
        )

        selected_feature = (
            forecast_resources[
                selected_resource
            ]["feature"]
        )

        selected_unit = (
            forecast_resources[
                selected_resource
            ]["unit"]
        )

        last_date = (
            forecast_history["Date"].max()
        )

        future_dates = pd.date_range(
            start=(
                last_date +
                pd.Timedelta(days=1)
            ),
            periods=7,
            freq="D"
        )

        historical_chart_df = pd.DataFrame({
            "Date":
                forecast_history["Date"],
            "Usage":
                forecast_history[
                    selected_feature
                ].astype(float).values,
            "Type":
                "Historical"
        })

        forecast_chart_df = pd.DataFrame({
            "Date":
                future_dates,
            "Usage":
                selected_result[
                    "predictions"
                ],
            "Type":
                "Forecast"
        })

        combined_forecast_df = pd.concat(
            [
                historical_chart_df,
                forecast_chart_df
            ],
            ignore_index=True
        )

        fig_forecast = px.line(
            combined_forecast_df,
            x="Date",
            y="Usage",
            color="Type",
            markers=True,
            title=(
                f"{selected_resource} — "
                "Historical vs 7-Day Forecast"
            )
        )

        fig_forecast.update_layout(
            xaxis_title="Date",
            yaxis_title=(
                f"Usage ({selected_unit})"
            ),
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True
        )

        st.subheader(
            "🔮 Next 7 Days Prediction"
        )

        forecast_table = pd.DataFrame({
            "Date":
                future_dates,
            "Predicted Usage":
                selected_result[
                    "predictions"
                ]
        })

        forecast_table[
            "Predicted Usage"
        ] = (
            forecast_table[
                "Predicted Usage"
            ].round(2)
        )

        st.dataframe(
            forecast_table,
            use_container_width=True,
            hide_index=True
        )

        trend_change = (
            selected_result[
                "trend_change"
            ]
        )

        if trend_change > 5:

            warning_level = (
                "🔴 High Attention"
            )

            warning_message = (
                f"{selected_resource} shows an "
                f"increasing historical trend of "
                f"{trend_change:+.1f}%. "
                "Preventive investigation is recommended."
            )

        elif trend_change > 2:

            warning_level = "🟡 Monitor"

            warning_message = (
                f"{selected_resource} shows a moderate "
                f"increasing trend of "
                f"{trend_change:+.1f}%. "
                "Continue monitoring."
            )

        elif trend_change < -5:

            warning_level = (
                "🟢 Positive Trend"
            )

            warning_message = (
                f"{selected_resource} shows a decreasing "
                f"trend of {abs(trend_change):.1f}%."
            )

        else:

            warning_level = "🟢 Stable"

            warning_message = (
                f"{selected_resource} is showing a "
                f"relatively stable trend of "
                f"{trend_change:+.1f}%."
            )

        st.subheader(
            "⚠️ AI Early-Warning Analysis"
        )

        st.metric(
            "Forecast Status",
            warning_level
        )

        st.write(
            warning_message
        )

        st.subheader(
            "🤖 AI Forecast Interpretation"
        )

        if trend_change > 5:

            interpretation = (
                f"The predictive analysis identifies "
                f"{selected_resource} as an increasing "
                "resource-consumption trend. If the "
                "current pattern continues, facilities "
                "teams should investigate the underlying "
                "drivers."
            )

        elif trend_change > 2:

            interpretation = (
                f"{selected_resource} shows a moderate "
                "upward trend. This is an early signal "
                "for continued monitoring and preventive "
                "action."
            )

        elif trend_change < -5:

            interpretation = (
                f"{selected_resource} shows a decreasing "
                "trend. If the pattern continues, "
                "consumption may remain lower than recent "
                "levels. Future observations should be "
                "used for validation."
            )

        else:

            interpretation = (
                f"{selected_resource} is relatively stable. "
                "No strong directional signal is currently "
                "detected."
            )

        st.success(
            interpretation
        )

        forecast_actions = {
            "Electricity":
                "Review HVAC schedules, lighting usage "
                "and high-load computing or laboratory equipment.",

            "Water":
                "Inspect taps, pipelines, storage systems "
                "and fixtures for possible leakage.",

            "Waste":
                "Review major waste sources, segregation "
                "and recycling processes.",

            "Paper":
                "Review printing activity and encourage "
                "digital documentation."
        }

        st.subheader(
            "💡 Recommended Preventive Action"
        )

        st.info(
            forecast_actions[
                selected_resource
            ]
        )

        forecast_export = forecast_table.copy()

        forecast_export[
            "Resource"
        ] = selected_resource

        forecast_export[
            "Unit"
        ] = selected_unit

        st.download_button(
            "⬇️ Download 7-Day Forecast CSV",
            data=forecast_export.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                f"{selected_resource.lower()}_"
                "7_day_forecast.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

        st.warning(
            "Responsible AI: Forecast values are "
            "model-based estimates derived from "
            "historical dataset patterns. They are not "
            "guaranteed future values."
        )


# =========================================================
# TAB 11 — PROFESSIONAL SUSTAINABILITY REPORT
# =========================================================

with tab11:

    st.header(
        "📄 Sustainability Report"
    )

    st.caption(
        "Generate a consolidated professional report "
        "containing campus performance, AI findings, "
        "actions, goals and forecast insights."
    )

    st.divider()

    report_date = pd.Timestamp.now().strftime(
        "%d %B %Y"
    )

    st.subheader(
        "📑 Report Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Overall Score",
            f"{overall_score}/100"
        )

    with col2:

        st.metric(
            "AI Alerts",
            total_alerts
        )

    with col3:

        st.metric(
            "Buildings",
            view["Building"].nunique()
        )

    with col4:

        st.metric(
            "Records",
            f"{len(view):,}"
        )

    st.write(
        f"**Report Date:** {report_date}"
    )

    st.write(
        "**Project:** AI-Powered Sustainable Campus "
        "Resource Advisor"
    )

    st.write(
        "**Primary SDG:** SDG 12 — Responsible "
        "Consumption and Production"
    )

    st.divider()

    # -----------------------------------------------------
    # EXECUTIVE SUMMARY
    # -----------------------------------------------------

    st.subheader(
        "📝 Executive Summary"
    )

    weakest_report_row = (
        performance_df.loc[
            performance_df["Score"].idxmin()
        ]
    )

    executive_summary = (
        f"The AI-Powered Sustainable Campus Resource "
        f"Advisor analyzes electricity, water, waste and "
        f"paper consumption using campus resource data, "
        f"machine-learning anomaly detection and "
        f"AI-assisted decision support. "
        f"The current prototype performance score is "
        f"{overall_score}/100. "
        f"The resource requiring the greatest attention "
        f"is {weakest_report_row['Resource']}, with a "
        f"performance score of "
        f"{weakest_report_row['Score']}/100. "
        f"The system detected {total_alerts} unusual "
        f"records in the selected dataset."
    )

    st.write(
        executive_summary
    )

    # -----------------------------------------------------
    # RESOURCE PERFORMANCE
    # -----------------------------------------------------

    st.subheader(
        "📊 Resource Performance"
    )

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # AI ANOMALIES
    # -----------------------------------------------------

    st.subheader(
        "🚨 AI Anomaly Analysis"
    )

    st.write(
        f"The machine-learning model identified "
        f"**{total_alerts}** unusual records."
    )

    st.write(
        f"High severity: **{high_alerts}**"
    )

    st.write(
        f"Medium severity: **{medium_alerts}**"
    )

    st.write(
        f"Low severity: **{low_alerts}**"
    )

    # -----------------------------------------------------
    # ACTION PLAN
    # -----------------------------------------------------

    st.subheader(
        "🎯 Recommended Action Plan"
    )

    report_action_df = action_df.drop(
        columns=["_rank"],
        errors="ignore"
    )

    st.dataframe(
        report_action_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # GOAL SUMMARY
    # -----------------------------------------------------

    st.subheader(
        "🌱 Sustainability Goal Summary"
    )

    st.write(
        f"Overall configured goal progress: "
        f"**{overall_goal_progress:.1f}%**."
    )

    # -----------------------------------------------------
    # FORECAST SUMMARY
    # -----------------------------------------------------

    st.subheader(
        "🔮 Seven-Day Forecast Summary"
    )

    forecast_summary_rows = []

    for resource_name, result in (
        forecast_results.items()
        if "forecast_results" in locals()
        else []
    ):

        forecast_summary_rows.append({
            "Resource":
                resource_name,
            "Trend":
                result["trend"],
            "Trend Change":
                f"{result['trend_change']:+.1f}%"
        })

    if forecast_summary_rows:

        forecast_summary_df = pd.DataFrame(
            forecast_summary_rows
        )

        st.dataframe(
            forecast_summary_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.write(
            "Forecast information is available in "
            "the Forecast section."
        )

    # -----------------------------------------------------
    # SDG ALIGNMENT
    # -----------------------------------------------------

    st.subheader(
        "🌍 SDG Alignment"
    )

    st.write(
        "**Primary SDG 12 — Responsible Consumption "
        "and Production:** The system supports responsible "
        "resource consumption by identifying unusual usage "
        "patterns and recommending practical corrective "
        "actions."
    )

    st.write(
        "**Supporting alignment:** Energy-related actions "
        "support SDG 7, while water-related actions support "
        "SDG 6."
    )

    # -----------------------------------------------------
    # RESPONSIBLE AI
    # -----------------------------------------------------

    st.subheader(
        "🔐 Responsible AI Statement"
    )

    st.write(
        "The system is designed as decision-support "
        "technology. AI-generated recommendations, "
        "anomaly findings and forecasts should not be "
        "treated as guaranteed facts or automatic "
        "operational decisions. Important findings should "
        "be verified by appropriate campus personnel. "
        "The prototype should avoid unnecessary personal "
        "information and should use understandable evidence "
        "such as observed usage, baselines and deviation "
        "percentages."
    )

    # -----------------------------------------------------
    # CREATE PROFESSIONAL PDF
    # -----------------------------------------------------

    def create_pdf_report():

        try:

            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
                PageBreak
            )

        except ImportError:

            return None, (
                "ReportLab is not installed. "
                "Run: pip install reportlab"
            )

        buffer = io.BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=16 * mm,
            leftMargin=16 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=22,
            leading=27,
            spaceAfter=8
        )

        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=11,
            leading=15,
            spaceAfter=15
        )

        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            spaceBefore=10,
            spaceAfter=7
        )

        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=14,
            spaceAfter=7
        )

        small_style = ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11
        )

        story = []

        story.append(
            Spacer(1, 18 * mm)
        )

        story.append(
            Paragraph(
                "CAMPUS SUSTAINABILITY REPORT",
                title_style
            )
        )

        story.append(
            Paragraph(
                "AI-Powered Sustainable Campus Resource Advisor",
                subtitle_style
            )
        )

        story.append(
            Paragraph(
                f"<b>Report Date:</b> {report_date}",
                body_style
            )
        )

        story.append(
            Paragraph(
                "<b>Primary SDG:</b> SDG 12 — "
                "Responsible Consumption and Production",
                body_style
            )
        )

        story.append(
            Spacer(1, 8 * mm)
        )

        # KPI TABLE

        kpi_data = [
            [
                "Overall Score",
                "AI Alerts",
                "Buildings",
                "Records"
            ],
            [
                f"{overall_score}/100",
                str(total_alerts),
                str(
                    view["Building"].nunique()
                ),
                f"{len(view):,}"
            ]
        ]

        kpi_table = Table(
            kpi_data,
            colWidths=[
                42 * mm,
                42 * mm,
                42 * mm,
                42 * mm
            ]
        )

        kpi_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f4e78")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, 1),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ])
        )

        story.append(
            kpi_table
        )

        story.append(
            Spacer(1, 8 * mm)
        )

        # EXECUTIVE SUMMARY

        story.append(
            Paragraph(
                "1. Executive Summary",
                heading_style
            )
        )

        story.append(
            Paragraph(
                executive_summary,
                body_style
            )
        )

        # RESOURCE PERFORMANCE

        story.append(
            Paragraph(
                "2. Resource Performance",
                heading_style
            )
        )

        resource_pdf_data = [
            [
                "Resource",
                "Score",
                "Recent Change",
                "Anomaly Rate",
                "Status"
            ]
        ]

        for _, row in performance_df.iterrows():

            resource_pdf_data.append([
                str(row["Resource"]),
                str(row["Score"]),
                str(row["Recent Change"]),
                str(row["Anomaly Rate"]),
                str(row["Status"])
            ])

        resource_table = Table(
            resource_pdf_data,
            colWidths=[
                35 * mm,
                25 * mm,
                32 * mm,
                32 * mm,
                45 * mm
            ],
            repeatRows=1
        )

        resource_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f4e78")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7.5
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ])
        )

        story.append(
            resource_table
        )

        # ANOMALY SECTION

        story.append(
            Paragraph(
                "3. AI Anomaly Analysis",
                heading_style
            )
        )

        anomaly_text = (
            f"The Isolation Forest model identified "
            f"{total_alerts} unusual records in the "
            f"selected dataset. Severity distribution: "
            f"{high_alerts} high, {medium_alerts} medium "
            f"and {low_alerts} low."
        )

        story.append(
            Paragraph(
                anomaly_text,
                body_style
            )
        )

        # ACTION PLAN

        story.append(
            Paragraph(
                "4. Recommended Action Plan",
                heading_style
            )
        )

        action_pdf_data = [
            [
                "Resource",
                "Priority",
                "Responsible Team",
                "Recommended Action"
            ]
        ]

        for _, row in (
            report_action_df.iterrows()
        ):

            action_pdf_data.append([
                str(row["Resource"]),
                str(row["Priority"]),
                str(row["Responsible Team"]),
                str(row["Recommended Action"])
            ])

        action_table = Table(
            action_pdf_data,
            colWidths=[
                28 * mm,
                25 * mm,
                37 * mm,
                80 * mm
            ],
            repeatRows=1
        )

        action_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f4e78")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ])
        )

        story.append(
            action_table
        )

        # GOALS

        story.append(
            Paragraph(
                "5. Sustainability Goals",
                heading_style
            )
        )

        story.append(
            Paragraph(
                f"Overall configured goal progress: "
                f"{overall_goal_progress:.1f}%.",
                body_style
            )
        )

        # FORECAST

        story.append(
            Paragraph(
                "6. Seven-Day Forecast",
                heading_style
            )
        )

        if forecast_summary_rows:

            forecast_pdf_data = [
                [
                    "Resource",
                    "Trend",
                    "Trend Change"
                ]
            ]

            for row in forecast_summary_rows:

                forecast_pdf_data.append([
                    row["Resource"],
                    row["Trend"],
                    row["Trend Change"]
                ])

            forecast_table_pdf = Table(
                forecast_pdf_data,
                colWidths=[
                    50 * mm,
                    60 * mm,
                    50 * mm
                ],
                repeatRows=1
            )

            forecast_table_pdf.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1f4e78")
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.grey
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    )
                ])
            )

            story.append(
                forecast_table_pdf
            )

        # SDG

        story.append(
            Paragraph(
                "7. SDG Alignment",
                heading_style
            )
        )

        story.append(
            Paragraph(
                "The primary alignment is SDG 12 — "
                "Responsible Consumption and Production. "
                "The project also supports SDG 7 through "
                "energy-efficiency actions and SDG 6 "
                "through water-conservation actions.",
                body_style
            )
        )

        # RESPONSIBLE AI

        story.append(
            Paragraph(
                "8. Responsible AI",
                heading_style
            )
        )

        story.append(
            Paragraph(
                "AI-generated findings are decision-support "
                "information and should be verified by "
                "responsible campus personnel. Forecasts "
                "are estimates rather than guaranteed future "
                "values. The system should minimize personal "
                "data collection and explain important findings "
                "using observable evidence such as resource "
                "usage, historical baselines and deviation "
                "percentages.",
                body_style
            )
        )

        story.append(
            Spacer(1, 8 * mm)
        )

        story.append(
            Paragraph(
                "AI-Powered Sustainable Campus Resource Advisor "
                "• IBM Granite 4.1 3B + Ollama",
                small_style
            )
        )

        document.build(
            story
        )

        buffer.seek(0)

        return (
            buffer.getvalue(),
            None
        )

    # -----------------------------------------------------
    # PDF GENERATION
    # -----------------------------------------------------

    if st.button(
        "📄 Generate Professional PDF Report",
        type="primary",
        use_container_width=True
    ):

        pdf_bytes, pdf_error = (
            create_pdf_report()
        )

        if pdf_bytes:

            st.success(
                "✅ Professional sustainability "
                "report generated successfully."
            )

            st.download_button(
                "📥 Download Professional PDF Report",
                data=pdf_bytes,
                file_name=(
                    "Campus_Sustainability_Report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

        else:

            st.error(
                pdf_error
            )

            st.code(
                "pip install reportlab",
                language="powershell"
            )

    st.warning(
        "Responsible AI: This report summarizes "
        "prototype decision-support results. Verify "
        "important findings before using them for "
        "real operational decisions."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="project-footer">
        AI-Powered Sustainable Campus Resource Advisor<br>
        IBM Granite 4.1 3B + Ollama • Responsible AI • SDG 12
    </div>
    """,
    unsafe_allow_html=True
)