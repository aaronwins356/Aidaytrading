"""Settings and administration page."""
from __future__ import annotations

import difflib
from pathlib import Path

import streamlit as st
import yaml

from data_io import CONFIG_PATH, load_config, save_config, seed_demo_data

st.set_page_config(page_title="Settings · Aurora Desk", page_icon="⚙️")

st.title("Settings & Admin")
config, raw = load_config(CONFIG_PATH)
st.session_state["config"] = config
st.session_state["config_raw"] = raw

st.subheader("Configuration editor")
with st.form("config_editor"):
    editable = yaml.safe_dump(raw or config.dict(), sort_keys=False, allow_unicode=True)
    text = st.text_area("YAML config", editable, height=400)
    submitted = st.form_submit_button("Validate & Save", use_container_width=True)

if submitted:
    try:
        new_data = yaml.safe_load(text) or {}
        success, message = save_config(new_data)
        if success:
            st.success(message)
            st.session_state["config"] = config.__class__(**new_data)
            st.session_state["config_raw"] = new_data
            diff = difflib.unified_diff(
                editable.splitlines(),
                yaml.safe_dump(new_data, sort_keys=False).splitlines(),
                lineterm="",
            )
            diff_text = "\n".join(diff) or "No changes"
            st.code(diff_text, language="diff")
        else:
            st.error(message)
    except yaml.YAMLError as exc:
        st.error(f"Invalid YAML: {exc}")

st.subheader("Kill switch")
if st.button("Flatten all positions (paper mode)"):
    Path("desk/kill_switch.request").write_text("flatten", encoding="utf-8")
    st.warning("Kill switch request written. Ensure risk engine picks this up.")

st.subheader("Job controls")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Refresh data cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Trigger a rerun (press r).")
with col2:
    if st.button("Recompute analytics"):
        st.cache_data.clear()
        st.success("Analytics cache will rebuild on next run.")
with col3:
    if st.button("Re-seed demo data"):
        seed_demo_data()
        st.success("Demo data seeded.")

st.subheader("Danger zone")
confirm = st.checkbox("I understand this cannot be undone")
if st.button("Wipe demo data", disabled=not confirm):
    for path in ["desk/db/paper_trading.sqlite", "desk/db/live_trading.sqlite"]:
        if Path(path).exists():
            Path(path).unlink()
    st.warning("Demo databases removed. Restart app to rebuild.")

st.subheader("Backups")
backup_dir = Path("desk/backups")
backup_dir.mkdir(parents=True, exist_ok=True)
if st.button("Backup config"):
    target = backup_dir / f"config_{Path(CONFIG_PATH).stem}.yaml"
    target.write_text(yaml.safe_dump(st.session_state["config_raw"], sort_keys=False), encoding="utf-8")
    st.success(f"Backup saved to {target}")

if st.button("Restore latest backup"):
    backups = sorted(backup_dir.glob("config_*.yaml"), reverse=True)
    if backups:
        latest = backups[0]
        data = yaml.safe_load(latest.read_text())
        success, message = save_config(data)
        if success:
            st.success(f"Restored from {latest}")
        else:
            st.error(message)
    else:
        st.error("No backups found.")
