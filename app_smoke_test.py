from pathlib import Path

from streamlit.testing.v1 import AppTest


app_path = Path(__file__).resolve().parent / "app.py"
app_test = AppTest.from_file(app_path, default_timeout=180)
app_test.run(timeout=180)

exceptions = [str(item) for item in app_test.exception]
if exceptions:
    raise AssertionError(f"Streamlit exceptions: {exceptions}")
if app_test.error:
    raise AssertionError("Streamlit displayed an error box.")

expected_tabs = [
    "Overview",
    "Scenario comparison",
    "Representative case",
    "Survey evidence",
    "Methods and limitations",
]
observed_tabs = [tab.label for tab in app_test.tabs]
if observed_tabs != expected_tabs:
    raise AssertionError(
        f"Unexpected tabs: {observed_tabs}"
    )

print("Streamlit AppTest: PASS")
print("Tabs:", observed_tabs)
