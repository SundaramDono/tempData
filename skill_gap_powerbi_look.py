import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

import openai

# Comment out the API key and related configurations
# Azure OpenAI API details - using environment variables for security
# AZURE_OPENAI_DEPLOYMENT = os.environ.get("DEPLOYMENT_NAME")
# AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

# Configure OpenAI client
# openai.api_type = "azure"
# openai.api_base = AZURE_OPENAI_ENDPOINT
# openai.api_key = AZURE_OPENAI_API_KEY
# openai.api_version = "2023-07-01-preview"

# Sample Skill Data (Replace with actual data)
skills_data = [
    {"Skill": "Business Intelligence", "Score": 95, "Expected": 100, "Gap": 5, "Sub-Skills": "KPI Development: 5%"},
    {"Skill": "Data Visualization", "Score": 92, "Expected": 100, "Gap": 8, "Sub-Skills": "Interactive Dashboards: 2%"},
    {"Skill": "Big Data Processing", "Score": 90, "Expected": 100, "Gap": 10, "Sub-Skills": "Distributed Computing: 3%"},
    {"Skill": "Scikit-learn (ML)", "Score": 93, "Expected": 100, "Gap": 7, "Sub-Skills": "Feature Engineering: 2%"},
    {"Skill": "Tableau", "Score": 96, "Expected": 100, "Gap": 4, "Sub-Skills": ""},
    {"Skill": "Machine Learning", "Score": 94, "Expected": 100, "Gap": 6, "Sub-Skills": "Regression Analysis: 3%"},
    {"Skill": "Data Management", "Score": 88, "Expected": 100, "Gap": 12, "Sub-Skills": "Data Cleaning: 5%, Missing Value Imputation: 2%, Data Transformation: 5%"},
    {"Skill": "Statistical Analysis", "Score": 91, "Expected": 100, "Gap": 9, "Sub-Skills": "Hypothesis Testing: 5%"},
    {"Skill": "Predictive Analytics", "Score": 89, "Expected": 100, "Gap": 11, "Sub-Skills": "Regression Analysis: 3%"},
    {"Skill": "Business Skills", "Score": 97, "Expected": 100, "Gap": 3, "Sub-Skills": "Domain Expertise: 1%, Communication Skills: 1%, Project Management: 1%, Ethics and Governance: 2%"},
]

# Function to Generate Insights from LLM
def fetch_insights(skill_data):
    # client = openai.AzureOpenAI(api_version="2023-07-01-preview")
    # prompt = f"""
    # Given the following skill assessment data:
    # {skill_data}

    # Provide:
    # 1. 🚨 Improvement Focus Areas (3-4 points, concise)
    # 2. 🛠️ Technical Tools Deficiency (2-3 points)
    # 3. 📉 Data Science Techniques Struggles (2-3 points)

    # Keep responses short, precise, and actionable.
    # """

    # response = client.chat.completions.create(
    # model=AZURE_OPENAI_DEPLOYMENT,
    # messages=[
    #     {"role": "system", "content": "You are an AI specialized in analyzing occupational standards and extracting detailed skill hierarchies."},
    #     {"role": "user", "content": prompt}
    # ],
    # temperature=0.1,
    # max_tokens=3000
    # )

    # return response.choices[0].message.content
    return "Insights generation is currently disabled."

skills_df = pd.DataFrame(skills_data)
marks_obtained = round(sum(skill["Score"] for skill in skills_data) / len(skills_data), 1)
total_marks = 100  # Assuming this is fixed

# Compute Marketability Score
industry_demand = {
    "Machine Learning": 0.9,
    "Big Data Processing": 0.85,
    "Tableau": 0.75,
    "Business Intelligence": 0.8,
    "Predictive Analytics": 0.88,
}

for skill in skills_data:
    demand_weight = industry_demand.get(skill["Skill"], 0.7)  # Default weight if not listed
    related_skills_score = sum(s["Score"] for s in skills_data if s["Skill"] != skill["Skill"]) / len(skills_data)
    relevance_weight = 1 - demand_weight  # Complementary weight

    skill["Marketable Score"] = round((skill["Score"] * demand_weight) + (related_skills_score * relevance_weight), 2)

# Convert to DataFrame for easier calculations
skills_df = pd.DataFrame(skills_data)
marketability_score = round(skills_df["Marketable Score"].mean(), 1)

# Marketability Status
marketability_status = "🔥 Highly Competitive" if marketability_score >= 80 else \
                       "✅ Competitive" if marketability_score >= 60 else \
                       "⚠️ Needs Improvement"

# Dashboard Theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# KPI Cards
kpi_cards = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H6("Marketability Score", className="text-white", style={'fontSize': '13px'}),
            html.H5(f"{marketability_score:.2f} ({marketability_status})", className="text-white",
                    style={'fontSize': '16px'})
        ])
    ], color="danger" if marketability_status == "🔥 Highly Competitive"
    else "#dc3545" if marketability_status == "✅ Competitive"
    else "warning" if marketability_status == "⚠️ Needs Improvement"
    else "secondary", inverse=True), width=3),

    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H6("Assessment Difficulty", className="text-white", style={'fontSize': '13px'}),
            html.H5("Easy (20%) | Medium (60%) | Difficult (20%)", className="text-white", style={'fontSize': '14px'})
        ])
    ], color="info", inverse=True), width=3),

    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H6("Total Marks", className="text-white", style={'fontSize': '13px'}),
            html.H5(str(total_marks), className="text-white", style={'fontSize': '16px'})
        ])
    ], color="primary", inverse=True), width=3),

    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H6("Marks Obtained", className="text-white", style={'fontSize': '13px', 'color':'#dc3545'}),
            html.H5(str(marks_obtained), className="text-white", style={'fontSize': '16px', 'color':'#dc3545'})
        ])
    ], color="success", inverse=True), width=3)
], className="mb-4")

# Assessment Overview (Compact)
assessment_overview = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H6("Assessment Overview", className="text-dark", style={'fontSize': '12px'}),
            html.H5("Job: Data Analytics Professional", className="text-primary", style={'fontSize': '14px'}) #Changed to 92 to reflect higher score
        ])
    ], color="light", inverse=False), width=3)
], className="mb-4")

# Radar Chart for Skill Gaps
fig_radar = px.line_polar(
    skills_df, r="Score", theta="Skill", line_close=True,
    title="Candidate Skill Profile"
).update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100]))
)

# Sunburst Chart for Hierarchical Skill Coverage
fig_sunburst = go.Figure(go.Sunburst(
    labels=["Data Management", "Statistical Analysis", "Hypothesis Testing", "Data Transformation",
            "Data Cleaning", "Missing Value Imputation", "Business Skills", "KPI Development",
            "Interactive Dashboards", "Predictive Analytics", "Machine Learning", "Feature Engineering"],

    parents=["", "Data Management", "Statistical Analysis", "Data Management",
             "Data Management", "Data Cleaning", "", "Business Skills",
             "Business Skills", "Business Skills", "Predictive Analytics", "Machine Learning"],

    branchvalues='total',
    marker=dict(colors=["#007bff", "#6610f2", "#6f42c1", "#e83e8c", "#dc3545", "#fd7e14",
                                    "#ffc107", "#28a745", "#20c997", "#17a2b8", "#6c757d", "#343a40"])
))

fig_sunburst.update_layout(title="Hierarchical Skill Coverage with Weight Distribution", paper_bgcolor='white', title_font_size=14)

# Skill Projection Trajectory Chart (Sample Data)
current_score = round(sum(skill["Score"] for skill in skills_data) / len(skills_data), 1)
average_gap = round(sum(skill["Gap"] for skill in skills_data) / len(skills_data), 1)
growth_rate = (average_gap / 3)  # Distribute the gap closure over 3 years

# Generate Future Projection
projection_df = pd.DataFrame({
    "Year": [2023, 2024, 2025, 2026],
    "Score": [
        current_score,  # 2023
        min(100, round(current_score + growth_rate, 1)),  # 2024
        min(100, round(current_score + 2 * growth_rate, 1)),  # 2025
        100  # 2026 (assuming max proficiency)
    ]
})

# Create Projection Figure
fig_projection = go.Figure(go.Scatter(
    x=projection_df["Year"],
    y=projection_df["Score"],
    mode='lines+markers',
    line=dict(color='blue', width=2),
    marker=dict(size=8, color='red')
))

# Update Layout
fig_projection.update_layout(
    title="Skill Projection Trajectory",
    xaxis_title="Year",
    yaxis_title="Score",
    paper_bgcolor='white',
    title_font_size=14
)

# Enhanced Table
def format_sub_skills(sub_skills):
    if not sub_skills:
        return "No sub-skills available"

    sub_skills_list = sub_skills.split(", ")
    return "\n".join(f"• {s.strip()}" for s in sub_skills_list)

for row in skills_data:
    row["Sub-Skills"] = format_sub_skills(row["Sub-Skills"])

# Define color conditions
color_conditions = [
    {
        "if": {"column_id": "Gap", "filter_query": "{Gap} <= 5"},
        "backgroundColor": "#b3e5fc",  # Light Blue
        "color": "black",
    },
    {
        "if": {"column_id": "Gap", "filter_query": "{Gap} > 5 && {Gap} <= 15"},
        "backgroundColor": "#64b5f6",  # Medium-Light Blue
        "color": "white",
    },
    {
        "if": {"column_id": "Gap", "filter_query": "{Gap} > 15 && {Gap} <= 30"},
        "backgroundColor": "#1976d2",  # Medium-Dark Blue
        "color": "white",
    },
    {
        "if": {"column_id": "Gap", "filter_query": "{Gap} > 30"},
        "backgroundColor": "#0d47a1",  # Dark Blue
        "color": "white",
    },
]

# Dash DataTable
skill_table = dash_table.DataTable(
    id="skill_table",
    columns=[
        {"name": "Skill", "id": "Skill"},
        {"name": "Score", "id": "Score"},
        {"name": "Expected", "id": "Expected"},
        {"name": "Gap (%)", "id": "Gap"},
        {"name": "Sub-Skills", "id": "Sub-Skills", "presentation": "markdown"}  # Enable bullet points
    ],
    data=skills_data,
    style_table={"width": "100%", "overflowX": "auto", "boxShadow": "0 0 10px rgba(0, 0, 0, 0.1)", "borderRadius": "5px"},
    style_cell={
        "textAlign": "left",
        "padding": "5px",  # Reduced padding from 10px to 5px
        "fontSize": "12px",  # Reduced font size slightly
        "color": "#212529",
        "border": "1px solid #dee2e6",
        "height": "20px",  # Set minimum row height
        "minHeight": "20px",  # Ensures rows stay compact
    },
    style_header={
        "backgroundColor": "#f8f9fa",
        "fontWeight": "bold",
        "borderBottom": "2px solid #dee2e6",
        "fontSize": "14px",
        "color": "#212529",
    },
    style_data_conditional=color_conditions + [
        {"if": {"column_id": "Sub-Skills"}, "whiteSpace": "pre-line"}  # Allow newlines
    ],
    page_size=10,
    filter_action="native",  # Search bar
    sort_action="native",  # Sorting
)

# Actionable Roadmap Summary
roadmap_summary = dbc.Card(
    dbc.CardBody([
        html.H6("Actionable Roadmap for Skill Enhancement", className="text-primary", style={'fontSize': '14px', 'fontWeight': 'bold'}),
        html.P([
            "Candidate exhibits exceptional proficiency across core data analytics skills. To further elevate their expertise, consider the following:",
            html.Ul([
                html.Li(html.Strong("Business Intelligence:")),
                html.P("📖 Take courses on KPI development and business reporting. Focus on Power BI & Tableau."),
                html.Li(html.Strong("Scikit-learn (ML):")),
                html.P("📊 Work on feature engineering & model selection. Try Kaggle challenges."),
                html.Li(html.Strong("Big Data Processing:")),
                html.P("⚙️ Hands-on with Hadoop, Spark & distributed computing projects."),
                html.Li(html.Strong("Data Visualization:")),
                html.P("🎨 Master interactive dashboards in Tableau/Power BI."),
                html.Li(html.Strong("Machine Learning:")),
                html.P("🤖 Implement real-world ML projects, focusing on regression & classification.")
            ]),
            "Leveraging their strong foundation, continuous learning and practical experience will ensure the candidate remains a leader in the dynamic field of data analytics."
        ], style={'fontSize': '12px'})
    ]),
    className="mt-2",
    style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'margin': '10px'}
)

# Fetch insights dynamically from Azure OpenAI
insights_text = fetch_insights(skills_data)

# Layout
app.layout = dbc.Container([
    dbc.Row([], style={'height': '20px'}),
    dbc.Row([
        dbc.Col(html.H1("Skill Gap Analysis Dashboard",
                        className="text-left text-secondary  fw-semibold",
                        style={'fontSize': '20px', 'marginBottom': '10px', 'font-family':'Nunito', 'color': '#333 !important', 'text-align' : 'left !important'}), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H5("Job Role: Data Analytics Professional",
                        className="text-left text-secondary",
                        style={'fontSize': '18px'}), width=12)
    ]),
    dbc.Row([], style={'height': '20px'}),
    kpi_cards,
    dbc.Row([
        dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        dcc.Graph(figure=fig_radar, style={'height': '500px', 'width': '650px'})
                    ]),
                    className="border shadow-sm",
                ), width=6
            ),

        dbc.Col(dcc.Graph(figure=fig_sunburst, style={'height': '525px', 'width': '700px'}), width=6),
    ]),
    dbc.Row([
        dbc.Col(dbc.Card(
            dbc.CardBody([
                html.H4("🔍 Key Observations & Insights", className="card-title text-black"),
                html.P(insights_text, className="card-text text-black", style={"white-space": "pre-line"}),
            ]),
            className="mt-4",
            inverse=True,
            style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'margin': '10px','fontSize': '12px', 'height': '400px', 'width': '650px'}
        ), className="mt-2",),
        dbc.Col(html.Div(roadmap_summary, style={'height': '500px', 'width': '650px'}), width=6),
    ]),
    dbc.Row([
        dbc.Col(skill_table, width=12),
    ]),
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True, port='8066')