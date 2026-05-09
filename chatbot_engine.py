import json
import os
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = os.path.join(os.path.dirname(__file__), "chatbot_data.json")


@lru_cache(maxsize=1)
def load_chatbot_index():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        entries = json.load(file)

    documents = [
        f"{entry['question']} {entry['answer']}"
        for entry in entries
    ]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents)
    return entries, vectorizer, matrix


def _format_recent_logs(logs):
    if not logs:
        return "No packets have been captured yet."
    return "\n".join(f"- {log}" for log in logs[-5:])


def _traffic_summary(context):
    total_count = context["total_count"]
    attack_count = context["attack_count"]
    normal_count = context["normal_count"]
    error_count = context["error_count"]

    if total_count == 0:
        return (
            "No live packets have been captured yet. Start IDS, then open a website "
            "or generate network activity to see traffic, normal packets, and alerts."
        )

    attack_percent = (attack_count / total_count) * 100
    normal_percent = (normal_count / total_count) * 100

    return (
        f"Current IDS traffic summary:\n"
        f"- Status: {context['status']}\n"
        f"- Total packets/logs: {total_count}\n"
        f"- Normal packets: {normal_count} ({normal_percent:.1f}%)\n"
        f"- Suspicious events: {attack_count} ({attack_percent:.1f}%)\n"
        f"- Processing errors: {error_count}\n"
        f"- Latest event: {context['latest_log']}"
    )


def _traffic_graph_summary(context):
    total_count = context["total_count"]
    if total_count == 0:
        return (
            "The traffic graph is empty because no packets have been captured yet. "
            "Start IDS and generate traffic to populate the graph."
        )

    return (
        "The current traffic graph compares normal packets, suspicious events, and errors. "
        f"Right now it should show Normal={context['normal_count']}, "
        f"Threat={context['attack_count']}, and Error={context['error_count']}."
    )


def _live_context_answer(question, context):
    normalized = question.lower()

    traffic_words = ["traffic", "packet", "packets", "logs", "network activity"]
    current_words = ["current", "live", "now", "running", "today", "this ids", "dashboard"]
    graph_words = ["graph", "chart", "overview", "bar chart", "visual"]

    if any(word in normalized for word in graph_words):
        return _traffic_graph_summary(context)

    if any(word in normalized for word in traffic_words) and any(word in normalized for word in current_words):
        return _traffic_summary(context)

    if any(phrase in normalized for phrase in ["tell traffic", "show traffic", "traffic summary", "ids traffic"]):
        return _traffic_summary(context)

    if any(word in normalized for word in ["status", "running", "stopped"]):
        return (
            f"The IDS is currently {context['status']}. It has processed "
            f"{context['total_count']} packets in this session."
        )

    if any(word in normalized for word in ["latest", "last", "recent log", "recent alert"]):
        return f"Recent IDS events:\n{_format_recent_logs(context['recent_logs'])}"

    if any(word in normalized for word in ["attack count", "threat count", "how many attack", "how many threat", "attacks", "threats"]):
        return (
            f"The IDS has detected {context['attack_count']} suspicious events "
            f"out of {context['total_count']} total packets. "
            f"Normal packets: {context['normal_count']}."
        )

    if any(word in normalized for word in ["normal count", "how many normal", "normal packets", "normal traffic"]):
        return (
            f"The IDS has recorded {context['normal_count']} normal packets "
            f"out of {context['total_count']} total packets. "
            f"Suspicious events: {context['attack_count']}."
        )

    if "error" in normalized:
        return f"The dashboard has recorded {context['error_count']} processing errors."

    return None


def _small_talk_answer(question):
    normalized = question.lower().strip()
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)

    greetings = {"hi", "hello", "hey", "hii", "good morning", "good evening"}
    if normalized in greetings:
        return (
            "Hello. I am your custom IDS chatbot. I can explain cybersecurity concepts, "
            "describe this IDS project, and answer questions about the live detection logs."
        )

    if "who are you" in normalized or "what can you do" in normalized:
        return (
            "I am a custom chatbot built for this IDS project. I use a local knowledge base "
            "and the current dashboard data, so I do not need OpenAI or Gemini."
        )

    if "thank" in normalized:
        return "You are welcome. Ask me about attacks, IDS logs, protocols, or the model."

    return None


def answer_question(question, context):
    question = question.strip()
    if not question:
        return "Please type a question about the IDS, cybersecurity, or live traffic."

    small_talk = _small_talk_answer(question)
    if small_talk:
        return small_talk

    live_answer = _live_context_answer(question, context)
    if live_answer:
        return live_answer

    entries, vectorizer, matrix = load_chatbot_index()
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).flatten()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    if best_score < 0.12:
        return (
            "I do not have a confident answer for that yet. Add this topic to "
            "chatbot_data.json to train my knowledge base further. I can currently "
            "answer IDS, packet, protocol, attack, model, and live log questions."
        )

    matched = entries[best_index]
    return matched["answer"]
