"""
Cocoa Procurement Agent - Streamlit app
Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud (add GEMINI_API_KEY / GROQ_API_KEY in Secrets)
"""
import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

import cocoa_agent_core as C

st.set_page_config(page_title="Cocoa Procurement Agent", page_icon="🍫", layout="wide")
MAX_RUNS_PER_SESSION = 5   # protects the shared free API quota


def secret(name):
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:
        return os.environ.get(name)


def md(text: str) -> str:
    """Escape $ so Streamlit does not treat prices as LaTeX."""
    return text.replace("$", "\\$")


@st.cache_resource(ttl=6 * 3600, show_spinner="Loading price, weather and news data...")
def get_data():
    return C.load_all_data(cache_dir="cache")


# ---------------- sidebar: parameters the user can change
with st.sidebar:
    st.header("Decision")
    today = dt.date.today()
    as_of = st.date_input("Decision date", value=today, min_value=dt.date(2024, 1, 1), max_value=today,
                          help="Pick a past date to test the agent against what actually happened next.")
    horizon = st.select_slider("Planning horizon (months)", options=[3, 6, 12], value=6)
    risk = st.radio("Risk appetite", ["conservative", "balanced", "aggressive"], index=1, horizontal=True)
    weather_view = st.selectbox("Weather assumption", [
        "Let the agent decide from the data",
        "Strong El Nino: drier West African cocoa belt",
        "Weak or no El Nino: near-normal weather"])

    st.header("Company (fictional)")
    tonnes = st.number_input("Cocoa need (tonnes/month)", 10.0, 5000.0, 120.0, 10.0)
    revenue = st.number_input("Revenue ($/month)", 100_000.0, 1e9, 3_000_000.0, 100_000.0)
    other = st.number_input("Other costs ($/month)", 0.0, 1e9, 1_500_000.0, 100_000.0)
    cur_hedge = st.slider("Already hedged (%)", 0, 100, 25)

    st.header("Agent")
    extra = st.multiselect("Data the agent may use (price is always on)",
                           ["weather", "news", "supply"], default=["weather", "news", "supply"])
    provider = st.selectbox("Model provider", ["Auto (Gemini, then Groq)", "Gemini only", "Groq only"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    runs = st.slider("Runs (consistency check)", 1, 3, 1)
    with st.expander("Use your own free API key (optional)"):
        user_gemini = st.text_input("Gemini API key", type="password")
        user_groq = st.text_input("Groq API key", type="password")
        st.caption("Keys stay in this session only. Get one at aistudio.google.com or console.groq.com.")

profile = C.CompanyProfile(monthly_cocoa_t=tonnes, monthly_revenue_usd=revenue,
                           monthly_other_costs_usd=other, current_hedge_pct=float(cur_hedge))
enabled = tuple(["price", "margin"] + extra)

# ---------------- header
st.title("🍫 Cocoa Procurement Agent")
st.caption("AI Agents in Business · Case 1: the cocoa price shock. A fictional UK chocolate maker asks how much "
           "of its next months' cocoa to buy forward now. Teaching demo - not investment advice.")

try:
    data = get_data()
except Exception as e:
    st.error(f"Could not load price data: {e}")
    st.stop()

with st.expander("Data status"):
    for k, v in data.status.items():
        st.write(f"**{k}**: {v}")
    st.write(f"**price series**: {data.price_source}")

snap = C.get_snapshot(data, as_of)
tools = C.AgentTools(snap, profile, horizon, enabled)
locked, locked_date, _ = tools.current_price()
rules = C.rules_baseline(tools)
actual = C.actual_prices_after(data, as_of, horizon)

col_chart, col_info = st.columns([2, 1])
with col_chart:
    st.pyplot(C.plot_price_history(data, [as_of.isoformat()], start="2023-01-01"))
with col_info:
    st.metric("Price on decision date", f"${locked:,.0f}/t", help=f"Last close on or before {locked_date}")
    st.metric("Rules baseline hedge", f"{rules['hedge_pct']:.0f}%")
    st.caption(md(" · ".join(rules["why"])))
    if len(actual):
        st.info(f"{len(actual)} month(s) of actual prices exist after this date, so the result can be checked.")
    else:
        st.info("Too recent to check against reality yet - the run can be saved and checked later.")

# ---------------- run
st.session_state.setdefault("runs_used", 0)
own_key = bool(user_gemini or user_groq)
limit_hit = (not own_key) and st.session_state["runs_used"] >= MAX_RUNS_PER_SESSION

if st.button("Run agent", type="primary", disabled=limit_hit):
    order = {"Auto (Gemini, then Groq)": ("gemini", "groq"), "Gemini only": ("gemini",),
             "Groq only": ("groq",)}[provider]
    keys = {"gemini": user_gemini or secret("GEMINI_API_KEY"), "groq": user_groq or secret("GROQ_API_KEY")}
    try:
        router = C.LLMRouter(order=order, api_keys=keys)
    except Exception as e:
        st.error(str(e))
        st.stop()
    cfg = C.AgentConfig(as_of=as_of.isoformat(), horizon_months=horizon, risk_appetite=risk,
                        weather_view=weather_view, temperature=temperature, enabled_tools=enabled)
    results = []
    for r in range(runs):
        with st.status(f"Run {r + 1}/{runs}: agent investigating...", expanded=False) as status:
            try:
                res = C.run_agent(data, profile, cfg, router, verbose=False)
                for t in res["trace"]:
                    st.write(f"step {t['step']} · `{t['tool']}` {t['args'] or ''}")
                status.update(label=f"Run {r + 1}/{runs}: done in {res['seconds']}s with {res['model']}",
                              state="complete")
                results.append(res)
            except Exception as e:
                status.update(label=f"Run {r + 1}/{runs} failed", state="error")
                st.error(str(e)[:500])
    if not own_key:
        st.session_state["runs_used"] += 1
    st.session_state["results"] = results
    st.session_state["results_key"] = (as_of.isoformat(), horizon)

if limit_hit:
    st.warning("Session run limit reached for the shared key. Add your own free API key in the sidebar to continue.")

# ---------------- results
results = st.session_state.get("results") or []
if results and st.session_state.get("results_key") == (as_of.isoformat(), horizon):
    res = results[0]
    rec = res["recommendation"]
    hedges = [x["recommendation"]["recommended_hedge_pct"] for x in results]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended hedge", f"{np.mean(hedges):.0f}%",
              help="Mean across runs" if len(hedges) > 1 else None)
    m2.metric("Confidence", rec["confidence"])
    m3.metric("Probability-weighted price", f"${res['expected_price_usd_t']:,}/t")
    ec = rec.get("_evidence_check", {})
    m4.metric("Evidence check", f"{len(ec.get('valid', []))}/{len(ec.get('cited', []))} ids valid")
    if len(hedges) > 1:
        st.caption(f"Consistency: hedge across runs = {hedges} (std {np.std(hedges):.1f} points)")
    if res["validation_notes"]:
        st.caption("Validation notes: " + "; ".join(res["validation_notes"]))

    tab_memo, tab_test, tab_ev, tab_trace = st.tabs(["CFO memo", "Check against reality", "Evidence", "Agent trace"])

    with tab_memo:
        st.markdown(md(C.render_memo(res)))

    with tab_test:
        if len(actual):
            strategies = {"No hedge (buy monthly)": 0.0, "Fixed 50%": 50.0, "Full hedge": 100.0,
                          "Rules baseline": rules["hedge_pct"], "AI agent": float(np.mean(hedges))}
            costs = {k: C.simulate_cost(actual, locked, h, profile.monthly_cocoa_t) for k, h in strategies.items()}
            best = min(costs["No hedge (buy monthly)"], costs["Full hedge"])
            table = pd.DataFrame([{"Strategy": k, "Hedge %": round(h),
                                   "Avg price paid ($/t)": round(costs[k] / (profile.monthly_cocoa_t * len(actual))),
                                   "Total cost ($)": round(costs[k]),
                                   "Extra cost vs best in hindsight ($)": round(costs[k] - best)}
                                  for k, h in strategies.items()])
            st.write(f"Actual average price over the next {len(actual)} month(s): **${actual.mean():,.0f}/t** "
                     f"(decision-date price ${locked:,.0f}/t).".replace("$", "\\$"))
            st.dataframe(table, hide_index=True, width="stretch")
            st.bar_chart(table.set_index("Strategy")["Avg price paid ($/t)"])
            st.caption("Simplification: the decision-date futures price is used as the forward price.")
        else:
            st.write("No actual prices after this date yet. Download the run and compare later.")
        st.subheader("Scenario margins at the recommended hedge")
        st.dataframe(pd.DataFrame(res["scenario_margins"]), hide_index=True, width="stretch")

    with tab_ev:
        ev = pd.DataFrame(res["evidence"])
        if not ev.empty:
            ev["cited"] = ev["id"].isin(ec.get("valid", []))
            st.dataframe(ev[["id", "cited", "type", "detail", "url"]], hide_index=True, width="stretch",
                         column_config={"url": st.column_config.LinkColumn("Source")})
        if ec.get("unsupported"):
            st.warning(f"Cited but not found in tool results: {ec['unsupported']}")

    with tab_trace:
        st.dataframe(pd.DataFrame(res["trace"]), hide_index=True, width="stretch")

    st.download_button("Download run (JSON)", json.dumps(results, indent=2, default=str),
                       file_name=f"cocoa_agent_run_{res['as_of']}.json", mime="application/json")
