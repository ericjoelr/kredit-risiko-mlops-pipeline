"""Streamlit-Oberfläche für die Credit Risk API. Reiner HTTP-Client: Validierung,
Modell und Schwellenwert liegen in der API. Start: streamlit run frontend/app.py"""

import math
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

PURPOSES = {
    "debt_consolidation": "Umschuldung",
    "credit_card": "Kreditkarte",
    "home_improvement": "Renovierung",
    "major_purchase": "Größere Anschaffung",
    "small_business": "Kleingewerbe",
    "educational": "Ausbildung",
    "all_other": "Sonstiges",
}

# Eingabefeld → (Label, Startwert, Minimum, Maximum, Schrittweite)
NUMBER_FIELDS = {
    "fico": ("FICO-Score", 710, 300, 850, 1),
    "annual_income": ("Jahreseinkommen (USD)", 60_000, 1_000, None, 1_000),
    "int_rate_pct": ("Zinssatz (%)", 12.0, 0.1, 99.0, 0.1),
    "installment": ("Monatliche Rate (USD)", 300.0, 0.0, None, 10.0),
    "dti": ("Schuldenquote (DTI)", 15.0, 0.0, None, 0.5),
    "credit_years": ("Kredithistorie (Jahre)", 12.0, 0.0, None, 0.5),
    "revol_bal": ("Offener Kreditrahmen-Saldo (USD)", 12_000, 0, None, 500),
    "revol_util": ("Auslastung Kreditrahmen (%)", 45.0, 0.0, 100.0, 1.0),
    "inq_last_6mths": ("Kreditanfragen letzte 6 Monate", 1, 0, None, 1),
    "delinq_2yrs": ("Zahlungsverzüge letzte 2 Jahre", 0, 0, None, 1),
    "pub_rec": ("Negative öffentliche Einträge", 0, 0, None, 1),
}


def to_application(form: dict) -> dict:
    """Formularwerte in nutzerfreundlichen Einheiten → Felder, die die API erwartet."""
    return {
        "credit_policy": int(form["credit_policy"]),
        "purpose": form["purpose"],
        "int_rate": form["int_rate_pct"] / 100,
        "installment": form["installment"],
        "log_annual_inc": math.log(form["annual_income"]),
        "dti": form["dti"],
        "fico": form["fico"],
        "days_with_cr_line": form["credit_years"] * 365.25,
        "revol_bal": form["revol_bal"],
        "revol_util": form["revol_util"],
        "inq_last_6mths": form["inq_last_6mths"],
        "delinq_2yrs": form["delinq_2yrs"],
        "pub_rec": form["pub_rec"],
    }


st.set_page_config(page_title="Kreditrisiko", layout="wide")
st.title("Kreditrisiko-Prüfung")

with st.sidebar:
    st.subheader("Modell")
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        st.write(f"**{info['model_type']}** · Version {info['version']}")
        st.write(f"Schwellenwert: {info['threshold']:.2f}")
        st.write(f"Test ROC-AUC: {info['test_metrics']['roc_auc']:.3f}")
    except requests.RequestException:
        st.error(f"API unter {API_URL} nicht erreichbar. Läuft sie (make run)?")

with st.form("application"):
    form = {
        "purpose": st.selectbox("Verwendungszweck", PURPOSES, format_func=PURPOSES.get),
        "credit_policy": st.checkbox("Erfüllt die Kreditrichtlinien", value=True),
    }
    columns = st.columns(3)
    for i, (key, (label, value, min_value, max_value, step)) in enumerate(NUMBER_FIELDS.items()):
        form[key] = columns[i % 3].number_input(label, value=value, min_value=min_value, max_value=max_value, step=step)
    submitted = st.form_submit_button("Risiko berechnen")

if submitted:
    try:
        response = requests.post(f"{API_URL}/predict", json=to_application(form), timeout=10)
    except requests.RequestException:
        st.error("API nicht erreichbar.")
        st.stop()

    if response.status_code == 422:  # Validierungsfehler der API anzeigen
        for error in response.json()["detail"]:
            st.error(f"{error['loc'][-1]}: {error['msg']}")
        st.stop()

    result = response.json()
    probability = result["default_probability"]
    if result["decision"] == "approve":
        st.success(f"Genehmigen – Ausfallwahrscheinlichkeit {probability:.1%}")
    else:
        st.error(f"Ablehnen – Ausfallwahrscheinlichkeit {probability:.1%}")

    history = st.session_state.setdefault("history", [])
    history.append(
        {
            "Zweck": PURPOSES[form["purpose"]],
            "FICO": form["fico"],
            "Ausfall": f"{probability:.1%}",
            "Entscheidung": result["decision"],
        }
    )

if st.session_state.get("history"):
    st.subheader("Bisherige Anfragen")
    st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True)
