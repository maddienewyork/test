import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.mark.parametrize("kind", ["continuous", "binary", "survival"])
@pytest.mark.parametrize("mode", ["Summary data", "Estimate + SE"])
def test_app_runs(kind, mode):
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    at.sidebar.selectbox[0].set_value(kind).run()
    [r for r in at.sidebar.radio if r.label == "Input"][0].set_value(mode).run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Power at planned N", "Probability of success", "Simulated LPI (median)"} <= labels
    if kind == "survival":
        assert "Final analysis (median)" in labels
