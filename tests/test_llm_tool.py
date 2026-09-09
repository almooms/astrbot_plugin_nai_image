import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import mcp

from astrbot.api.event import MessageChain
from astrbot.api.message_components import Image, Plain
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.provider.entities import LLMResponse, ProviderRequest, TokenUsage
from astrbot.core.provider.provider import Provider
from main import NAIGenerateImagePlugin


class FakeEvent:
    def __init__(self):
        self.extras = {}
        self.sent_messages = []

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    async def send(self, message):
        self.sent_messages.append(message)

    def get_result(self):
        return None


def make_plugin(generate_result):
    plugin = object.__new__(NAIGenerateImagePlugin)
    plugin.call_mode = "direct"
    plugin.enable_llm_tool = True
    plugin.enable_tool_style = True
    plugin.image_gen_key = "test-token"
    plugin.image_style = "vertical"
    plugin.image_size = "竖图"
    plugin._generate_one = AsyncMock(return_value=generate_result)
    return plugin


def make_openai_plugin(generate_result):
    plugin = make_plugin(generate_result)
    plugin.call_mode = "openai"
    plugin.openai_api_base_url = "https://openai.example"
    plugin.openai_api_key = "sk-test"
    plugin.custom_artists = ""
    img_bytes, reason = generate_result
    plugin._openai_generate = AsyncMock(
        return_value=([img_bytes] if img_bytes else [], reason)
    )
    return plugin


async def collect_results(generator):
    return [result async for result in generator]


class ToolThenReplyProvider(Provider):
    def __init__(self):
        super().__init__({"id": "test-provider", "model": "test-model"}, {})
        self.call_count = 0

    def get_current_key(self):
        return "test-key"

    def set_key(self, key):
        return None

    async def get_models(self):
        return ["test-model"]

    async def text_chat(self, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            return LLMResponse(
                role="assistant",
                completion_text="",
                tools_call_name=["NAI_Generate_Image"],
                tools_call_args=[
                    {"prompt": "1girl", "style": "anime", "size_cn": "方图"}
                ],
                tools_call_ids=["call_nai_image"],
                usage=TokenUsage(input_other=10, output=5),
            )
        return LLMResponse(
            role="assistant",
            completion_text="图片已经发给你了。",
            usage=TokenUsage(input_other=10, output=5),
        )


class RepeatedToolThenReplyProvider(ToolThenReplyProvider):
    async def text_chat(self, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            return LLMResponse(
                role="assistant",
                completion_text="",
                tools_call_name=["NAI_Generate_Image", "NAI_Generate_Image"],
                tools_call_args=[
                    {"prompt": "1girl", "style": "anime", "size_cn": "方图"},
                    {"prompt": "2girls", "style": "galgame", "size_cn": "横图"},
                ],
                tools_call_ids=["call_nai_image_1", "call_nai_image_2"],
                usage=TokenUsage(input_other=10, output=5),
            )
        return LLMResponse(
            role="assistant",
            completion_text="图片已经发给你了。",
            usage=TokenUsage(input_other=10, output=5),
        )


def test_llm_tool_sends_image_and_returns_one_text_result():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()

    results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "方图"))
    )

    assert len(results) == 1
    assert results[0] == "图片已生成并发送给用户，请根据本次请求继续回复。"
    assert len(event.sent_messages) == 1
    assert isinstance(event.sent_messages[0], MessageChain)
    assert isinstance(event.sent_messages[0].chain[0], Plain)
    assert isinstance(event.sent_messages[0].chain[1], Image)
    assert event.sent_messages[0].chain[1].file.startswith("base64://")
    plugin._generate_one.assert_awaited_once_with("1girl", "anime", "方图")


def test_llm_tool_passes_supported_sizes_to_api_unchanged():
    for size_cn in (
        "竖图",
        "横图",
        "方图",
        "2K竖图",
        "2K横图",
        "2K方图",
        "4K竖图",
        "4K横图",
        "4K方图",
    ):
        plugin = make_plugin((b"test-image-bytes", "ok"))
        event = FakeEvent()

        results = asyncio.run(
            collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", size_cn))
        )

        assert results == ["图片已生成并发送给用户，请根据本次请求继续回复。"]
        plugin._generate_one.assert_awaited_once_with("1girl", "anime", size_cn)


def test_llm_tool_openai_mode_uses_pixel_resolution():
    plugin = make_openai_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()

    results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "1024x1536"))
    )

    assert results == ["图片已生成并发送给用户，请根据本次请求继续回复。"]
    args, _ = plugin._openai_generate.await_args
    assert args[1] == "1024x1536"


def test_llm_tool_openai_mode_maps_tier_names_to_resolution():
    plugin = make_openai_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()

    results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "4K竖图"))
    )

    assert results == ["图片已生成并发送给用户，请根据本次请求继续回复。"]
    args, _ = plugin._openai_generate.await_args
    # 4K 档在 OpenAI 接口被降级为 2K 上限分辨率（与 _openai_size 契约一致）
    assert args[1] == "1088x1920"


def test_llm_tool_openai_mode_rejects_invalid_size():
    plugin = make_openai_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()

    results = asyncio.run(
        collect_results(
            plugin.NAI_Generate_Image(event, "1girl", "anime", "not-a-size")
        )
    )

    assert len(results) == 1
    assert "未知尺寸" in results[0]
    assert "832x1216" in results[0]
    plugin._openai_generate.assert_not_awaited()


def test_llm_tool_direct_mode_maps_pixel_size_to_tier():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()

    results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "1024x1536"))
    )

    assert results == ["图片已生成并发送给用户，请根据本次请求继续回复。"]
    plugin._generate_one.assert_awaited_once_with("1girl", "anime", "竖图")


def test_llm_tool_defaults_style_and_size_from_plugin_config():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    plugin.image_style = "galgame"
    plugin.image_size = "2K横图"
    event = FakeEvent()

    results = asyncio.run(collect_results(plugin.NAI_Generate_Image(event, "1girl")))

    assert results == ["图片已生成并发送给用户，请根据本次请求继续回复。"]
    plugin._generate_one.assert_awaited_once_with("1girl", "galgame", "2K横图")


def test_llm_tool_style_disabled_forces_plugin_default_style():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    plugin.enable_tool_style = False
    plugin.image_style = "galgame"
    event = FakeEvent()

    # 即使模型仍然传来 style，开关关闭时也必须使用插件设置的默认画风
    results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "方图"))
    )

    assert results == ["图片已生成并发送给用户，请根据本次请求继续回复。"]
    plugin._generate_one.assert_awaited_once_with("1girl", "galgame", "方图")


def test_refresh_llm_tool_schema_hides_style_and_switches_size_format():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    registered = {
        "prompt": {"type": "string", "description": "提示词"},
        "style": {"type": "string", "description": "画风"},
        "size_cn": {"type": "string", "description": "尺寸"},
    }

    class FakeTool:
        def __init__(self):
            self.parameters = {"type": "object", "properties": dict(registered)}

    class FakeManager:
        def __init__(self):
            self.tool = FakeTool()

        def get_func(self, name):
            return self.tool if name == "NAI_Generate_Image" else None

    class FakeContext:
        def __init__(self):
            self.manager = FakeManager()

        def get_llm_tool_manager(self):
            return self.manager

    plugin.context = FakeContext()
    tool = plugin.context.get_llm_tool_manager().tool

    # 关闭画风参数 + OpenAI 模式：style 隐藏，size 描述改为像素分辨率
    plugin.enable_tool_style = False
    plugin.call_mode = "openai"
    plugin.image_size = "竖图"
    plugin._refresh_llm_tool_schema()
    props = tool.parameters["properties"]
    assert "style" not in props
    assert "宽x高" in props["size_cn"]["description"]

    # 重新开启 + 切回 direct：从原始 schema 重建，style 恢复，size 描述回到分档名
    plugin.enable_tool_style = True
    plugin.call_mode = "direct"
    plugin._refresh_llm_tool_schema()
    props = tool.parameters["properties"]
    assert "style" in props
    assert props["style"]["description"] == "画风"
    assert "竖图" in props["size_cn"]["description"]


def test_llm_tool_allows_only_one_generation_attempt_per_event():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()

    first_results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "方图"))
    )
    second_results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "2girls", "galgame", "横图"))
    )

    assert len(first_results) == 1
    assert first_results == ["图片已生成并发送给用户，请根据本次请求继续回复。"]
    assert len(second_results) == 1
    assert "本轮消息已经执行过一次" in second_results[0]
    assert "send_message_to_user" in second_results[0]
    assert len(event.sent_messages) == 1
    plugin._generate_one.assert_awaited_once()


def test_llm_tool_does_not_retry_after_failed_attempt_in_same_event():
    plugin = make_plugin((None, "timeout"))
    event = FakeEvent()

    first_results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "方图"))
    )
    second_results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "方图"))
    )

    assert len(first_results) == 1
    assert "生成失败" in first_results[0]
    assert len(second_results) == 1
    assert "本轮消息已经执行过一次" in second_results[0]
    assert event.sent_messages == []
    plugin._generate_one.assert_awaited_once()


def test_llm_tool_blocks_concurrent_generation_in_same_event():
    plugin = make_plugin((b"unused", "ok"))
    event = FakeEvent()
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_generate(*args):
        started.set()
        await release.wait()
        return b"test-image-bytes", "ok"

    plugin._generate_one = AsyncMock(side_effect=delayed_generate)

    async def run_calls():
        first_task = asyncio.create_task(
            collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "方图"))
        )
        await started.wait()
        second_results = await collect_results(
            plugin.NAI_Generate_Image(event, "2girls", "galgame", "横图")
        )
        release.set()
        first_results = await first_task
        return first_results, second_results

    first_results, second_results = asyncio.run(run_calls())

    assert first_results == ["图片已生成并发送给用户，请根据本次请求继续回复。"]
    assert len(second_results) == 1
    assert "本轮消息已经执行过一次" in second_results[0]
    assert len(event.sent_messages) == 1
    plugin._generate_one.assert_awaited_once()


def test_llm_tool_validation_error_does_not_consume_event_attempt():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()

    invalid_results = asyncio.run(
        collect_results(
            plugin.NAI_Generate_Image(event, "1girl", "unknown-style", "方图")
        )
    )
    valid_results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "方图"))
    )

    assert len(invalid_results) == 1
    assert "未知风格" in invalid_results[0]
    assert valid_results == ["图片已生成并发送给用户，请根据本次请求继续回复。"]
    assert len(event.sent_messages) == 1
    plugin._generate_one.assert_awaited_once()


def test_llm_tool_executor_returns_content_instead_of_ending_agent_loop():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()
    run_context = ContextWrapper(context=SimpleNamespace(event=event))
    tool = FunctionTool(
        name="NAI_Generate_Image",
        description="Generate an image",
        parameters={"type": "object", "properties": {}},
        handler=plugin.NAI_Generate_Image,
    )

    async def execute_tool():
        return [
            result
            async for result in FunctionToolExecutor.execute(
                tool,
                run_context,
                prompt="1girl",
                style="anime",
                size_cn="方图",
            )
        ]

    results = asyncio.run(execute_tool())

    assert len(results) == 1
    assert isinstance(results[0], mcp.types.CallToolResult)
    assert results[0].content[0].text == (
        "图片已生成并发送给用户，请根据本次请求继续回复。"
    )
    assert len(event.sent_messages) == 1


def test_llm_tool_keeps_agent_loop_open_and_preserves_complete_turn():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()
    provider = ToolThenReplyProvider()
    tool = FunctionTool(
        name="NAI_Generate_Image",
        description="Generate an image",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "style": {"type": "string"},
                "size_cn": {"type": "string"},
            },
        },
        handler=plugin.NAI_Generate_Image,
    )
    request = ProviderRequest(
        prompt="request 1",
        contexts=[],
        func_tool=ToolSet(tools=[tool]),
    )
    run_context = ContextWrapper(context=SimpleNamespace(event=event))
    runner = ToolLoopAgentRunner()

    async def run_agent():
        await runner.reset(
            provider=provider,
            request=request,
            run_context=run_context,
            tool_executor=FunctionToolExecutor(),
            agent_hooks=BaseAgentRunHooks(),
            streaming=False,
        )
        return [response async for response in runner.step_until_done(3)]

    asyncio.run(run_agent())

    assert provider.call_count == 2
    assert runner.done()
    assert runner.get_final_llm_resp().completion_text == "图片已经发给你了。"
    assert [message.role for message in runner.run_context.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert runner.run_context.messages[1].tool_calls[0].id == "call_nai_image"
    assert runner.run_context.messages[2].tool_call_id == "call_nai_image"
    assert len(event.sent_messages) == 1


def test_llm_tool_returns_results_for_repeated_calls_but_generates_once():
    plugin = make_plugin((b"test-image-bytes", "ok"))
    event = FakeEvent()
    provider = RepeatedToolThenReplyProvider()
    tool = FunctionTool(
        name="NAI_Generate_Image",
        description="Generate an image",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "style": {"type": "string"},
                "size_cn": {"type": "string"},
            },
        },
        handler=plugin.NAI_Generate_Image,
    )
    request = ProviderRequest(
        prompt="request repeated images",
        contexts=[],
        func_tool=ToolSet(tools=[tool]),
    )
    run_context = ContextWrapper(context=SimpleNamespace(event=event))
    runner = ToolLoopAgentRunner()

    async def run_agent():
        await runner.reset(
            provider=provider,
            request=request,
            run_context=run_context,
            tool_executor=FunctionToolExecutor(),
            agent_hooks=BaseAgentRunHooks(),
            streaming=False,
        )
        return [response async for response in runner.step_until_done(3)]

    asyncio.run(run_agent())

    assert provider.call_count == 2
    assert [message.role for message in runner.run_context.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert [
        tool_call.id for tool_call in runner.run_context.messages[1].tool_calls
    ] == ["call_nai_image_1", "call_nai_image_2"]
    assert [
        message.tool_call_id
        for message in runner.run_context.messages
        if message.role == "tool"
    ] == ["call_nai_image_1", "call_nai_image_2"]
    assert len(event.sent_messages) == 1
    plugin._generate_one.assert_awaited_once_with("1girl", "anime", "方图")


def test_llm_tool_returns_one_error_result():
    plugin = make_plugin((None, "timeout"))
    event = FakeEvent()

    results = asyncio.run(
        collect_results(plugin.NAI_Generate_Image(event, "1girl", "anime", "方图"))
    )

    assert len(results) == 1
    assert isinstance(results[0], str)
    assert "生成失败" in results[0]
    assert "超时" in results[0]
