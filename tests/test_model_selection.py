from rag_agent.agent.agent import OfflineRagAgent, build_agent
from rag_agent.config import Settings
from rag_agent.llm.models import OfflineOpenAIModel, build_openai_text_model


def test_model_helper_resolves_fast_role_name() -> None:
    settings = Settings(openai_model_default="gpt-5", openai_model_fast="gpt-5-mini")

    model = build_openai_text_model(settings, role="fast")

    assert isinstance(model, OfflineOpenAIModel)
    assert model.model_name == "gpt-5-mini"
    assert model.role == "fast"


def test_build_agent_uses_resolved_default_model_name() -> None:
    settings = Settings(openai_model="gpt-5-legacy", openai_model_default="gpt-5")

    agent = build_agent(settings)

    assert isinstance(agent, OfflineRagAgent)
    assert agent.model_name == "gpt-5"
