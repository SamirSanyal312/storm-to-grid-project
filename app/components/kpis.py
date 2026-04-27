"""Reusable KPI display helpers."""

import streamlit as st


def metric(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label=label, value=value, delta=delta)
