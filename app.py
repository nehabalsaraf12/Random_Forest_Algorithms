import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
MODEL_FILE = "random.pkl"

def load_model(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model(MODEL_FILE)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market & Employment Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 20px;
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .header-title {
            color: #38bdf8;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .header-subtitle {
            color: #94a3b8;
            font-size: 1.0rem;
        }
        .form-section {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .section-header {
            color: #818cf8;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 15px;
            border-bottom: 1px solid #334155;
            padding-bottom: 5px;
        }
        .grid-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 15px;
        }
        .input-group {
            display: flex;
            flex-direction: column;
        }
        .input-group label {
            font-size: 0.9rem;
            color: #cbd5e1;
            margin-bottom: 6px;
            font-weight: 500;
        }
        .input-group input {
            background-color: #0f172a;
            border: 1px solid #475569;
            border-radius: 8px;
            color: #f8fafc;
            padding: 10px 14px;
            font-size: 0.95rem;
        }
        .input-group input:focus {
            border-color: #38bdf8;
            outline: none;
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }
        .submit-btn {
            background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 14px 24px;
            border-radius: 8px;
            border: none;
            width: 100%;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            margin-top: 25px;
        }
        .submit-btn:hover {
            background: linear-gradient(90deg, #4f46e5 0%, #4338ca 100%);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5);
            transform: translateY(-1px);
        }
        .result-card {
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            margin-top: 25px;
        }
        .status-bull {
            background: linear-gradient(135deg, #065f46 0%, #047857 100%);
            border: 1px solid #10b981;
            color: #a7f3d0;
        }
        .status-recession {
            background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
            border: 1px solid #ef4444;
            color: #fecaca;
        }
        .status-stable {
            background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
            border: 1px solid #3b82f6;
            color: #bfdbfe;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metric-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #94a3b8;
        }
        .metric-value {
            font-size: 1.2rem;
            font-weight: 700;
            color: #f8fafc;
            margin-top: 5px;
        }
        .error-box {
            background-color: #7f1d1d;
            border: 1px solid #ef4444;
            color: #fecaca;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header-card">
            <div class="header-title">📊 Economic & Workforce Status Predictor</div>
            <div class="header-subtitle">Enter company, workforce, and macroeconomic metrics below to run the classification model.</div>
        </div>

        {% if error_message %}
        <div class="error-box">⚠️ {{ error_message }}</div>
        {% endif %}

        <div class="form-section">
            <form method="POST">
                <div class="section-header">🏢 Company & Timeline</div>
                <div class="grid-row">
                    <div class="input-group">
                        <label>Company Name ID</label>
                        <input type="number" name="company_name" value="{{ form_data.get('company_name', 1) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Industry Code</label>
                        <input type="number" name="industry" value="{{ form_data.get('industry', 1) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Country Code</label>
                        <input type="number" name="country" value="{{ form_data.get('country', 1) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Company Size</label>
                        <input type="number" name="company_size" value="{{ form_data.get('company_size', 500) }}" required>
                    </div>
                </div>
                <div class="grid-row">
                    <div class="input-group">
                        <label>Month (1-12)</label>
                        <input type="number" min="1" max="12" name="month" value="{{ form_data.get('month', 6) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Year</label>
                        <input type="number" name="year" value="{{ form_data.get('year', 2026) }}" required>
                    </div>
                </div>

                <div class="section-header">📉 Workforce & Layoff Metrics</div>
                <div class="grid-row">
                    <div class="input-group">
                        <label>Layoffs Count</label>
                        <input type="number" name="layoffs_count" value="{{ form_data.get('layoffs_count', 0) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Layoff Percentage (%)</label>
                        <input type="number" step="0.1" min="0" max="100" name="layoff_percentage" value="{{ form_data.get('layoff_percentage', 0.0) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Reason for Layoffs Code</label>
                        <input type="number" name="reason_for_layoffs" value="{{ form_data.get('reason_for_layoffs', 0) }}" required>
                    </div>
                </div>
                <div class="grid-row">
                    <div class="input-group">
                        <label>Open Roles Count</label>
                        <input type="number" name="open_roles" value="{{ form_data.get('open_roles', 15) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Hiring Trend Index</label>
                        <input type="number" name="hiring_trend" value="{{ form_data.get('hiring_trend', 5) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Remote Jobs (%)</label>
                        <input type="number" step="0.1" min="0" max="100" name="remote_jobs_percentage" value="{{ form_data.get('remote_jobs_percentage', 35.0) }}" required>
                    </div>
                </div>

                <div class="section-header">🤖 AI Automation & Sentiment</div>
                <div class="grid-row">
                    <div class="input-group">
                        <label>AI Automation Impact (0-10)</label>
                        <input type="number" step="0.1" min="0" max="10" name="ai_automation_impact" value="{{ form_data.get('ai_automation_impact', 3.5) }}" required>
                    </div>
                    <div class="input-group">
                        <label>AI Replacement Risk (0-10)</label>
                        <input type="number" step="0.1" min="0" max="10" name="ai_replacement_risk" value="{{ form_data.get('ai_replacement_risk', 2.1) }}" required>
                    </div>
                    <div class="input-group">
                        <label>AI Adoption Level (0-10)</label>
                        <input type="number" step="0.1" min="0" max="10" name="ai_adoption_level" value="{{ form_data.get('ai_adoption_level', 5.0) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Top Hiring Role Code</label>
                        <input type="number" name="top_hiring_role" value="{{ form_data.get('top_hiring_role', 10) }}" required>
                    </div>
                </div>

                <div class="section-header">💹 Financial Performance & Scores</div>
                <div class="grid-row">
                    <div class="input-group">
                        <label>Stock Growth (%)</label>
                        <input type="number" step="0.1" name="stock_growth_percent" value="{{ form_data.get('stock_growth_percent', 4.5) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Revenue Growth (%)</label>
                        <input type="number" step="0.1" name="revenue_growth_percent" value="{{ form_data.get('revenue_growth_percent', 6.2) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Salary Budget Change (%)</label>
                        <input type="number" step="0.1" name="salary_budget_change" value="{{ form_data.get('salary_budget_change', 2.0) }}" required>
                    </div>
                    <div class="input-group">
                        <label>Employee Sentiment (0-10)</label>
                        <input type="number" step="0.1" min="0" max="10" name="employee_sentiment" value="{{ form_data.get('employee_sentiment', 6.5) }}" required>
                    </div>
                </div>
                <div class="grid-row">
                    <div class="input-group">
                        <label>Job Security Score (0-10)</label>
                        <input type="number" step="0.1" min="0" max="10" name="job_security_score" value="{{ form_data.get('job_security_score', 7.0) }}" required>
                    </div>
                </div>

                <button type="submit" class="submit-btn">🚀 Run Classification Model</button>
            </form>
        </div>

        {% if prediction %}
            {% if prediction == "Bull Market" %}
                {% set css_class = "status-bull" %}
                {% set icon = "🟢" %}
            {% elif prediction == "Recession" %}
                {% set css_class = "status-recession" %}
                {% set icon = "🔴" %}
            {% else %}
                {% set css_class = "status-stable" %}
                {% set icon = "🔵" %}
            {% endif %}

            <div class="result-card {{ css_class }}">
                <h2>{{ icon }} Predicted Market Status: {{ prediction }}</h2>
                <h4 style="margin-top: 20px; color: inherit;">📊 Prediction Confidence Breakdown</h4>
                <div class="metrics-grid">
                    {% for prob in probabilities %}
                    <div class="metric-box">
                        <div class="metric-label">{{ prob.class }}</div>
                        <div class="metric-value">{{ prob.percent }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None
    error_message = None
    form_data = {}

    if model is None:
        error_message = f"Model file '{MODEL_FILE}' was not found in the root directory."

    if request.method == "POST":
        form_data = request.form
        if model is not None:
            try:
                input_data = np.array([[
                    float(form_data.get("company_name", 1)),
                    float(form_data.get("industry", 1)),
                    float(form_data.get("country", 1)),
                    float(form_data.get("company_size", 500)),
                    float(form_data.get("month", 6)),
                    float(form_data.get("year", 2026)),
                    float(form_data.get("layoffs_count", 0)),
                    float(form_data.get("layoff_percentage", 0.0)),
                    float(form_data.get("reason_for_layoffs", 0)),
                    float(form_data.get("ai_automation_impact", 3.5)),
                    float(form_data.get("ai_replacement_risk", 2.1)),
                    float(form_data.get("open_roles", 15)),
                    float(form_data.get("hiring_trend", 5)),
                    float(form_data.get("remote_jobs_percentage", 35.0)),
                    float(form_data.get("top_hiring_role", 10)),
                    float(form_data.get("stock_growth_percent", 4.5)),
                    float(form_data.get("revenue_growth_percent", 6.2)),
                    float(form_data.get("salary_budget_change", 2.0)),
                    float(form_data.get("ai_adoption_level", 5.0)),
                    float(form_data.get("employee_sentiment", 6.5)),
                    float(form_data.get("job_security_score", 7.0))
                ]])

                prediction = model.predict(input_data)[0]
                raw_probs = model.predict_proba(input_data)[0]
                classes = model.classes_
                probabilities = [{"class": cls, "percent": f"{p * 100:.2f}%"} for cls, p in zip(classes, raw_probs)]

            except Exception as e:
                error_message = f"An error occurred during prediction: {e}"

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        probabilities=probabilities,
        error_message=error_message,
        form_data=form_data
    )

if __name__ == "__main__":
    app.run(debug=True)