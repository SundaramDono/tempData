import dash
from dash import dcc, html, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from dash_bootstrap_components import themes
from numpy import sort

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[themes.BOOTSTRAP])
app.title = "Skill Assessment Dashboard"
# Load skills data from file
data_analytics_skills = {
    "job_title": "Data Analytics Professional",
    "description": "A Data Analytics Professional is responsible for collecting, cleaning, analyzing, and visualizing data to derive insights and support decision-making processes.",
    "skill_hierarchy": {
        "Technical Skills": {
            "Data Management": ["Data Cleaning", "Data Transformation"],
            "Statistical Analysis": ["Hypothesis Testing"],
            "Data Visualization": ["Interactive Dashboards"],
            "Predictive Analytics": ["Regression Analysis"],
            "Machine Learning": ["Feature Engineering"],
            "Big Data Processing": ["Distributed Computing"],
            "Business Intelligence": ["KPI Development"]
        },
        "Business Skills": ["Domain Expertise", "Communication Skills", "Project Management", "Ethics and Governance"]
    }
}
# Sample individual data
skills_data = [
    {"Skill": "Business Intelligence", "Score": 80, "Expected": 100, "Gap": 20, "Sub-Skills": "KPI Development: 10%"},
    {"Skill": "Data Visualization", "Score": 82, "Expected": 100, "Gap": 18,
     "Sub-Skills": "Interactive Dashboards: 8%"},
    {"Skill": "Big Data Processing", "Score": 78, "Expected": 100, "Gap": 22,
     "Sub-Skills": "Distributed Computing: 7%"},
    {"Skill": "Scikit-learn (ML)", "Score": 83, "Expected": 100, "Gap": 17, "Sub-Skills": "Feature Engineering: 6%"},
    {"Skill": "Tableau", "Score": 85, "Expected": 100, "Gap": 15, "Sub-Skills": ""},
    {"Skill": "Machine Learning", "Score": 79, "Expected": 100, "Gap": 21, "Sub-Skills": "Regression Analysis: 7%"},
    {"Skill": "Data Management", "Score": 80, "Expected": 100, "Gap": 20,
     "Sub-Skills": "Data Cleaning: 10%, Missing Value Imputation: 5%, Data Transformation: 5%"},
    {"Skill": "Statistical Analysis", "Score": 81, "Expected": 100, "Gap": 19, "Sub-Skills": "Hypothesis Testing: 9%"},
    {"Skill": "Predictive Analytics", "Score": 79, "Expected": 100, "Gap": 21, "Sub-Skills": "Regression Analysis: 6%"},
    {"Skill": "Business Skills", "Score": 84, "Expected": 100, "Gap": 16,
     "Sub-Skills": "Domain Expertise: 4%, Communication Skills: 4%, Project Management: 4%, Ethics and Governance: 4%"},
]

# Create sample data for multiple candidates
candidates = ["Alice", "Bob", "Charlie", "David", "Eve"]
candidate_data = []

# Generate sample data for multiple candidates with some random variation
for candidate in candidates:
    for skill in skills_data:
        # Add some random variation to scores
        random_adjustment = np.random.randint(-10, 10)
        score = min(100, max(60, skill["Score"] + random_adjustment))
        gap = skill["Expected"] - score

        candidate_data.append({
            "Candidate": candidate,
            "Skill": skill["Skill"],
            "Score": score,
            "Expected": skill["Expected"],
            "Gap": gap,
            "Sub-Skills": skill["Sub-Skills"]
        })

# Create DataFrame for candidate data
df_candidates = pd.DataFrame(candidate_data)

# Create sample data for candidate summary as shown in your example
candidate_summary = pd.DataFrame({
    "Candidate": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Total Skills Evaluated": [26, 27, 27, 27, 27],
    "Avg Proficiency": [67.5, 77.5, 72.5, 72.5, 75],
    "Avg Industry Benchmark": [82.5, 87.5, 81.5, 84, 82.5],
    "Avg Skill Gap (%)": [15, 10, 9, 11.5, 7.5]
})
df_candidates["Total Skills Evaluated"] = len(skills_data)
# Dynamically infer department skills from skills_data
# Infer departments dynamically (cycling between 'Analytics', 'IT', 'Business')
departments = ["Analytics", "IT", "Business"]
department_assignments = [departments[i % len(departments)] for i in range(len(skills_data))]

# Create department_skills dynamically from skills_data
department_skills = pd.DataFrame({
    "Department": department_assignments,
    "Skill": [skill["Skill"] for skill in skills_data],
    "Avg Proficiency": [skill["Score"] for skill in skills_data],
    "Industry Benchmark": [skill["Expected"] for skill in skills_data],
    "Avg Skill Gap (%)": sort([skill["Gap"] for skill in skills_data])
})

department_skills_sorted = department_skills.sort_values(by="Avg Skill Gap (%)", ascending=False)

# Sample role-based skill distribution
# Dynamically infer skills from skills_data
# Dynamically infer skills from skills_data
skills_list = [skill["Skill"] for skill in skills_data]
print("lenght::",len(skills_list))
print(len([75, 82, 78, np.nan, np.nan, 80, np.nan, 79, 84]))
# Ensure all roles have the same number of values as the skills list
role_skills = pd.DataFrame({
    "Skill": skills_list,
    "Data Analyst": [75, 82, 78, np.nan, np.nan, 80, np.nan, 79, 84,91],
    "Data Scientist": [np.nan, np.nan, 78, 83, 79, np.nan, 81, 79, np.nan,np.nan],
    "Engineer": [np.nan, 82, np.nan, np.nan, 79, 80, np.nan, np.nan, np.nan, 72]
})



# Sample proficiency vs experience data
experience_data = []
for candidate in candidates:
    # Random experience between 1 and 10 years
    experience = np.random.randint(1, 11)
    # Calculate average proficiency from candidate_data
    proficiency = df_candidates[df_candidates["Candidate"] == candidate]["Score"].mean()
    experience_data.append({
        "Candidate": candidate,
        "Experience (Years)": experience,
        "Avg Proficiency": proficiency
    })

df_experience = pd.DataFrame(experience_data)

# Sample historical data (before and after training)
periods = ["2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4", "2024 Q1", "2024 Q2"]
historical_data = []

# Generate sample historical data for each skill
for skill in [item["Skill"] for item in skills_data]:
    # Start with a base proficiency and gradually increase it
    base_proficiency = np.random.randint(60, 75)
    for i, period in enumerate(periods):
        # Increase proficiency over time (with some randomness)
        proficiency = min(95, base_proficiency + i * 3 + np.random.randint(-2, 3))
        historical_data.append({
            "Period": period,
            "Skill": skill,
            "Proficiency": proficiency
        })


# Create skill hierarchy visualization (Sunburst Chart)
def create_sunburst_chart():
    labels, parents = [], []
    for category, subcategories in data_analytics_skills["skill_hierarchy"].items():
        labels.append(category)
        parents.append("")
        if isinstance(subcategories, dict):
            for sub, subskills in subcategories.items():
                labels.append(sub)
                parents.append(category)
                for skill in subskills:
                    labels.append(skill)
                    parents.append(sub)
        else:
            for skill in subcategories:
                labels.append(skill)
                parents.append(category)

    fig = go.Figure(go.Sunburst(labels=labels, parents=parents, marker=dict(colors=px.colors.qualitative.Set2)))
    fig.update_layout(title="Skill Hierarchy for Data Analytics Job Profile")
    return fig

df_historical = pd.DataFrame(historical_data)

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("Skill Assessment Dashboard", className="mt-4 mb-4 text-center"),
        html.P("Interactive dashboard for analyzing skill assessments across candidates, departments, and roles",
               className="text-center mb-4")
    ], className="container"),



    html.Div([
        html.H2("1. Candidate Skill Comparison", className="mt-4 mb-3"),
        html.P("Overview of candidate performance against industry benchmarks"),
        dash_table.DataTable(
            id='candidate-summary',
            columns=[{"name": i, "id": i} for i in candidate_summary.columns],
            data=candidate_summary.to_dict('records'),
            style_table={"width": "100%", "overflowX": "auto", "boxShadow": "0 0 10px rgba(0, 0, 0, 0.1)",
                         "borderRadius": "5px"},
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
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            page_size=10,
            filter_action="native",  # Search bar
            sort_action="native",  # Sorting
        ),

        html.Div([
            html.Div([
                dcc.Graph(figure=create_sunburst_chart())
            ], className="col-md-6"),

            html.Div([
                dcc.Graph(
                    id='candidate-gap-chart',
                    figure=px.bar(
                        candidate_summary,
                        x="Candidate",
                        y=["Avg Proficiency", "Avg Industry Benchmark"],
                        barmode="group",
                        title="Proficiency vs Benchmark Comparison",
                        labels={"value": "Score", "variable": "Metric"},
                        color_discrete_sequence=["#66c2a5", "#8da0cb"],  # Custom Set2 colors (teal & blue)
                    ).update_layout(
                        yaxis_range=[0, 100],
                        legend_title="Metric",
                        bargap=0.2  # Adjust bar spacing for better readability
                    )
                )
            ], className="col-md-6")
            ,

        ], className="row mb-4"),

        html.H2("2. Department-Wise Skill Gap Analysis", className="mt-4 mb-3"),
        html.P("Identifies high-priority skills that need upskilling"),
        dash_table.DataTable(
            id='department-skills',
            columns=[{"name": i, "id": i} for i in department_skills.columns],
            data=department_skills.to_dict('records'),
            style_table={"width": "100%", "overflowX": "auto", "boxShadow": "0 0 10px rgba(0, 0, 0, 0.1)",
                         "borderRadius": "5px"},
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
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                    'if': {'filter_query': '{Avg Skill Gap (%)} > 15'},
                    'backgroundColor': '#ffcccc',
                    'color': 'black'
                }
            ],
            page_size=10,
            filter_action="native",  # Search bar
            sort_action="native",  # Sorting
        ),
        # html.Div([
        #             dcc.Graph(
        #                 id='department-gap-chart',
        #                 figure=px.bar(
        #                     department_skills,
        #                     x="Skill",
        #                     y="Avg Skill Gap (%)",
        #                     color="Department",
        #                     title="Department Skill Gaps",
        #                     color_discrete_sequence=px.colors.qualitative.Set1
        #                 ).update_layout(yaxis_range=[0, 30])
        #             )
        #         ], className="mb-4 mt-3"),

        html.Div([
            dcc.Graph(
                id='department-gap-chart',
                figure=px.bar(
                    department_skills_sorted,
                    x="Skill",
                    y="Avg Skill Gap (%)",
                    color="Department",
                    title="Department Skill Gaps (Stacked View)",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    barmode="stack"  # Stacked bars for better comparison
                ).update_layout(yaxis_range=[0, 30])

            )
        ], className="mb-4 mt-3"),

        html.H2("3. Role-Based Skill Distribution", className="mt-4 mb-3"),
        html.P("Heatmap showing proficiency levels per role"),
        html.Div([
            dcc.Graph(
                id='role-heatmap',
                figure=px.imshow(
                    role_skills.set_index('Skill'),
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    title="Role vs Skill Proficiency Heatmap"
                ).update_layout(
                    xaxis_title="Role",
                    yaxis_title="Skill"
                )
            )
        ], className="mb-4"),

        html.H2("4. Proficiency vs. Experience Analysis", className="mt-4 mb-3"),
        html.P("Identifies if experience contributes to skill mastery"),
        html.Div([
            dcc.Graph(
                id='experience-scatter',
                figure=px.scatter(
                    df_experience,
                    x="Experience (Years)",
                    y="Avg Proficiency",
                    color="Candidate",
                    size="Avg Proficiency",
                    hover_data=["Candidate"],
                    title="Proficiency vs Experience Correlation",
                    color_discrete_sequence=px.colors.qualitative.Set1
                ).update_layout(
                    xaxis_range=[0, 11],
                    yaxis_range=[50, 100]
                ).add_shape(
                    type="line",
                    x0=0, y0=75,
                    x1=10, y1=85,
                    line=dict(color="green", dash="dash")
                ).add_annotation(
                    x=5, y=87,
                    text="Industry Benchmark Trend",
                    showarrow=False,
                    font=dict(color="green")
                )
            )
        ], className="mb-4"),

        html.H2("5. Trend Analysis Over Time", className="mt-4 mb-3"),
        html.P("ROI measurement for training programs"),
        html.Div([
            dcc.Dropdown(
                id='skill-selector',
                options=[{'label': skill, 'value': skill} for skill in df_historical['Skill'].unique()],
                value=df_historical['Skill'].unique()[0],
                clearable=False,
                className="mb-3"
            ),
            dcc.Graph(id='historical-trend')
        ], className="mb-4"),

        html.Div([
            html.H2("6. Individual Skill Assessment", className="mt-4 mb-3"),
            html.P("Detailed view of an individual's skill assessment"),
            dcc.Dropdown(
                id='candidate-selector',
                options=[{'label': candidate, 'value': candidate} for candidate in df_candidates['Candidate'].unique()],
                value='Alice',
                clearable=False,
                className="mb-3"
            ),
            html.Div([
                    dcc.Graph(
                        id='candidate-radar-chart',
                        figure=px.line_polar(
                            df_candidates[df_candidates["Candidate"] == "Alice"],
                            r="Score",
                            theta="Skill",
                            line_close=True,
                            title="Candidate Skill Profile"
                        ).update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100])
                            )
                        )
                    )], className="col-md-6"),
            html.Div([
                dcc.Graph(id='individual-skills',
                      figure=px.bar(
                          color_discrete_sequence=px.colors.qualitative.Set2  # Matching the theme
                      )
                    )
                ], className="col-md-6"),

        ], className="row mb-4"),

    ], className="container"),

    html.Footer([
        html.P("Skills Assessment Dashboard © 2025", className="text-center mt-4 mb-4")
    ])
])


# Callbacks
@app.callback(
    Output('historical-trend', 'figure'),
    [Input('skill-selector', 'value')]
)
def update_historical_trend(selected_skill):
    filtered_df = df_historical[df_historical['Skill'] == selected_skill]

    fig = px.line(
        filtered_df,
        x='Period',
        y='Proficiency',
        title=f"Historical Trend for {selected_skill}",
        markers=True
    )

    # Add benchmark line
    fig.add_hline(
        y=85,
        line_dash="dash",
        line_color="red",
        annotation_text="Industry Benchmark",
        annotation_position="top right"
    )

    # Highlight training periods
    fig.add_vrect(
        x0="2023 Q2", x1="2023 Q3",
        fillcolor="green", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Training Period",
        annotation_position="top left"
    )

    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Proficiency Score",
        yaxis_range=[50, 100]
    )

    return fig


@app.callback(
    Output('individual-skills', 'figure'),
    [Input('candidate-selector', 'value')]
)
def update_individual_skills(selected_candidate):
    filtered_df = df_candidates[df_candidates['Candidate'] == selected_candidate]

    fig = go.Figure()

    # Add bar chart for scores (using Set2 color)
    fig.add_trace(go.Bar(
        x=filtered_df['Skill'],
        y=filtered_df['Score'],
        name='Current Proficiency',
        marker_color='#66c2a5'  # Teal (from Set2)
    ))

    # Add line for expected scores (using a non-red color)
    fig.add_trace(go.Scatter(
        x=filtered_df['Skill'],
        y=filtered_df['Expected'],
        name='Expected Proficiency',
        mode='lines+markers',
        marker=dict(color='#8da0cb'),  # Blueish shade from Set2
        line=dict(color='#8da0cb', dash='dash')  # Dashed line for distinction
    ))

    fig.update_layout(
        title=f"Skill Assessment for {selected_candidate}",
        xaxis_title='Skills',
        yaxis_title='Proficiency Score',
        yaxis_range=[0, 110],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig



@app.callback(
    Output('candidate-radar-chart', 'figure'),
    [Input('candidate-selector', 'value')]
)
def update_radar_chart(selected_candidate):
    filtered_df = df_candidates[df_candidates['Candidate'] == selected_candidate]

    fig = px.line_polar(
        filtered_df,
        r="Score",
        theta="Skill",
        line_close=True,
        title=f"{selected_candidate}'s Skill Profile"
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        )
    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)