from rag_agent.tui.app import build_tui_app


def test_build_tui_app_returns_app_object() -> None:
    app = build_tui_app()

    assert app is not None
