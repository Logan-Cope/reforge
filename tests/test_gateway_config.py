"""LLM gateway routing: use the AI Gateway when its key is set, else direct Anthropic."""
from reforge.llm.claude import _client_config


def test_direct_when_no_gateway_key():
    key, base_url = _client_config("sk-ant-x", "", "https://ai-gateway.vercel.sh")
    assert key == "sk-ant-x"
    assert base_url is None


def test_gateway_when_key_set():
    key, base_url = _client_config("sk-ant-x", "vck-y", "https://ai-gateway.vercel.sh")
    assert key == "vck-y"
    assert base_url == "https://ai-gateway.vercel.sh"


def test_gateway_key_without_url_falls_back_to_direct():
    key, base_url = _client_config("sk-ant-x", "vck-y", "")
    assert key == "sk-ant-x"
    assert base_url is None
