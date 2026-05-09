import os
import queue
import threading
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scapy.all import sniff
from streamlit_autorefresh import st_autorefresh

from chatbot_engine import answer_question, load_chatbot_index


warnings.filterwarnings("ignore", category=UserWarning)


st.set_page_config(
    page_title="Real-Time IDS",
    page_icon="IDS",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --ids-bg: #f6f8fb;
            --ids-panel: #ffffff;
            --ids-border: #d8e0ea;
            --ids-text: #172033;
            --ids-muted: #607086;
            --ids-accent: #0f766e;
            --ids-danger: #b42318;
            --ids-warning: #b54708;
        }

        .stApp {
            background: var(--ids-bg);
            color: var(--ids-text);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        [data-testid="stSidebar"] {
            background: #111827;
        }

        [data-testid="stSidebar"] * {
            color: #f9fafb;
        }

        .hero {
            border: 1px solid var(--ids-border);
            border-radius: 8px;
            padding: 22px 24px;
            background: linear-gradient(135deg, #ffffff 0%, #eef7f5 100%);
            margin-bottom: 18px;
        }

        .hero h1 {
            font-size: 2.15rem;
            line-height: 1.2;
            margin: 0 0 8px 0;
            color: var(--ids-text);
            letter-spacing: 0;
        }

        .hero p {
            color: var(--ids-muted);
            margin: 0;
            font-size: 1rem;
        }

        div[data-testid="metric-container"] {
            background: var(--ids-panel);
            border: 1px solid var(--ids-border);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: none;
        }

        div[data-testid="metric-container"] * {
            color: var(--ids-text);
        }

        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] p {
            color: inherit;
        }

        .log-row {
            border: 1px solid var(--ids-border);
            border-left-width: 5px;
            border-radius: 8px;
            padding: 11px 13px;
            margin-bottom: 9px;
            background: var(--ids-panel);
            color: var(--ids-text);
            overflow-wrap: anywhere;
        }

        .log-normal {
            border-left-color: var(--ids-accent);
        }

        .log-threat {
            border-left-color: var(--ids-danger);
        }

        .log-error {
            border-left-color: var(--ids-warning);
        }

        .section-label {
            color: var(--ids-muted);
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            text-transform: uppercase;
        }

        .stChatMessage {
            border: 1px solid var(--ids-border);
            border-radius: 8px;
            background: #ffffff;
        }

        .stChatMessage,
        .stChatMessage *,
        div[data-testid="stChatMessage"],
        div[data-testid="stChatMessage"] * {
            color: var(--ids-text) !important;
        }

        div[data-testid="stChatMessage"] {
            background: #ffffff;
            border: 1px solid var(--ids-border);
            border-radius: 8px;
        }

        div[data-testid="stChatMessage"] code,
        div[data-testid="stChatMessage"] pre {
            color: #f9fafb !important;
        }

        @media (max-width: 700px) {
            .hero {
                padding: 18px;
            }

            .hero h1 {
                font-size: 1.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()


@st.cache_resource
def load_models():
    saved_data = joblib.load("model/model.pkl")
    if isinstance(saved_data, dict):
        return (
            saved_data["model"],
            saved_data.get("label_encoder"),
            saved_data.get("features"),
            saved_data.get("anomaly_model"),
        )
    return saved_data, None, None, None


model, le, feature_cols, iso_model = load_models()


if "logs" not in st.session_state:
    st.session_state.logs = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hi, I am your IDS assistant. Ask me about the live traffic, "
                "security concepts, attack counts, or what the latest alert means."
            ),
        }
    ]


@st.cache_resource
def get_packet_queue():
    return queue.Queue()


packet_queue = get_packet_queue()


@st.cache_resource
def get_shared_state():
    return {"running": False, "started_at": None}


shared_state = get_shared_state()


def classify_log(log):
    if log.startswith("OK"):
        return "normal"
    if log.startswith("Error"):
        return "error"
    return "threat"


def process_packet(packet):
    try:
        if feature_cols is not None:
            length = len(packet)
            proto = packet.proto if hasattr(packet, "proto") else 0
            src_port = packet.sport if hasattr(packet, "sport") else 0
            dst_port = packet.dport if hasattr(packet, "dport") else 0

            tcp = 1 if packet.haslayer("TCP") else 0
            udp = 1 if packet.haslayer("UDP") else 0
            icmp = 1 if packet.haslayer("ICMP") else 0

            features = [[length, proto, src_port, dst_port, tcp, udp, icmp]]

            if hasattr(model, "n_features_in_") and len(features[0]) != model.n_features_in_:
                n_features = model.n_features_in_
                row = features[0]
                features = [row + [0] * (n_features - len(row)) if len(row) < n_features else row[:n_features]]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                pred_idx = model.predict(features)[0]

            if iso_model is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    anomaly = iso_model.predict(features)[0]
            else:
                anomaly = 1

            if anomaly == -1:
                label = "high threat anomaly"
            elif pred_idx == 0:
                label = "normal"
            else:
                label = "low threat"
        else:
            features = [
                len(packet),
                1 if packet.haslayer("IP") else 0,
                1 if packet.haslayer("TCP") else 0,
                1 if packet.haslayer("UDP") else 0,
            ]
            full_features = np.zeros((1, model.n_features_in_))
            n_assign = min(len(features), model.n_features_in_)
            full_features[0, :n_assign] = features[:n_assign]
            if hasattr(model, "feature_names_in_"):
                full_features = pd.DataFrame(full_features, columns=model.feature_names_in_)
            pred_idx = model.predict(full_features)[0]
            label = "normal" if pred_idx in [0, "normal"] else f"attack {pred_idx}"

        summary = packet.summary()
        timestamp = time.strftime("%H:%M:%S")

        if label.lower() == "normal":
            result = f"OK {timestamp} | Normal traffic | {summary}"
        else:
            result = f"ALERT {timestamp} | Suspicious activity: {label.upper()} | {summary}"

        packet_queue.put(result)
    except Exception as exc:
        packet_queue.put(f"Error: {exc}")


def stop_filter(packet):
    return not shared_state["running"]


def sniff_loop():
    sniff(prn=process_packet, store=False, stop_filter=stop_filter)


def get_ids_context():
    logs = st.session_state.logs
    attack_count = sum(1 for log in logs if classify_log(log) == "threat")
    normal_count = sum(1 for log in logs if classify_log(log) == "normal")
    error_count = sum(1 for log in logs if classify_log(log) == "error")
    latest_log = logs[-1] if logs else "No packets have been captured yet."
    status = "running" if shared_state["running"] else "stopped"

    return {
        "status": status,
        "attack_count": attack_count,
        "normal_count": normal_count,
        "error_count": error_count,
        "total_count": len(logs),
        "latest_log": latest_log,
        "recent_logs": logs[-8:],
    }


def ask_chatbot(prompt):
    return answer_question(prompt, get_ids_context())


with st.sidebar:
    st.markdown("### IDS Control")
    st.caption("Start or stop live packet monitoring.")

    start_clicked = st.button("Start IDS", use_container_width=True, type="primary")
    stop_clicked = st.button("Stop IDS", use_container_width=True)

    st.divider()
    st.markdown("### View")
    selected_view = st.radio(
        "Choose dashboard view",
        ["Live Monitor", "AI Assistant", "Model Info"],
        label_visibility="collapsed",
        key="selected_view",
    )

    st.divider()
    st.markdown("### Assistant Mode")
    st.success("Custom chatbot ready")
    st.caption("Uses chatbot_data.json. No API key required.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. What would you like to check next?",
            }
        ]
        st.rerun()


if selected_view == "Live Monitor":
    st_autorefresh(interval=2000, key="refresh")


if start_clicked and not shared_state["running"]:
    shared_state["running"] = True
    shared_state["started_at"] = time.time()
    thread = threading.Thread(target=sniff_loop, daemon=True)
    thread.start()

if stop_clicked:
    shared_state["running"] = False


while not packet_queue.empty():
    st.session_state.logs.append(packet_queue.get())


context = get_ids_context()

st.markdown(
    """
    <div class="hero">
        <h1>Real-Time AI Intrusion Detection System</h1>
        <p>Monitor packets, review live threat signals, and ask the assistant about your network activity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Status", context["status"].title())
metric_cols[1].metric("Threats", context["attack_count"])
metric_cols[2].metric("Normal", context["normal_count"])
metric_cols[3].metric("Total Packets", context["total_count"])

if selected_view == "Live Monitor":
    left_col, right_col = st.columns([1.25, 0.75], gap="large")

    with left_col:
        st.markdown('<div class="section-label">Live Detection Logs</div>', unsafe_allow_html=True)
        if not st.session_state.logs:
            st.info("No packets yet. Start IDS, then open a website to generate traffic.")
        else:
            for log in reversed(st.session_state.logs[-25:]):
                log_type = classify_log(log)
                css_class = {
                    "normal": "log-normal",
                    "threat": "log-threat",
                    "error": "log-error",
                }[log_type]
                st.markdown(
                    f'<div class="log-row {css_class}">{log}</div>',
                    unsafe_allow_html=True,
                )

    with right_col:
        st.markdown('<div class="section-label">Traffic Overview</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame(
            {
                "Traffic": ["Normal", "Threat", "Error"],
                "Count": [
                    context["normal_count"],
                    context["attack_count"],
                    context["error_count"],
                ],
            }
        )
        if context["total_count"] == 0:
            st.info("Traffic chart will appear after packets are captured.")
        else:
            st.bar_chart(chart_df.set_index("Traffic"))

        if context["latest_log"] != "No packets have been captured yet.":
            st.markdown('<div class="section-label">Latest Event</div>', unsafe_allow_html=True)
            st.code(context["latest_log"])


elif selected_view == "AI Assistant":
    st.markdown('<div class="section-label">AI Security Assistant</div>', unsafe_allow_html=True)

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about this IDS, your alerts, or any security topic")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = ask_chatbot(prompt)
            st.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})


elif selected_view == "Model Info":
    st.markdown('<div class="section-label">Detection Pipeline</div>', unsafe_allow_html=True)
    st.write(
        "The dashboard captures packets with Scapy, converts them into model features, "
        "classifies traffic with the trained model, and shows live results in the monitor."
    )

    model_cols = st.columns(3)
    model_cols[0].metric("Feature Count", len(feature_cols) if feature_cols else getattr(model, "n_features_in_", "Unknown"))
    model_cols[1].metric("Anomaly Model", "Enabled" if iso_model is not None else "Disabled")
    model_cols[2].metric("Assistant", "Custom")

    if feature_cols:
        st.markdown('<div class="section-label">Model Features</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Feature": feature_cols}), use_container_width=True)

    st.markdown('<div class="section-label">Chatbot Knowledge Base</div>', unsafe_allow_html=True)
    entries, _, _ = load_chatbot_index()
    st.metric("Custom Q&A Entries", len(entries))
    st.dataframe(pd.DataFrame(entries), use_container_width=True)
