from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import re
import random

import aiohttp
from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter, MessageChain
from astrbot.api.message_components import Record, Reply, Plain
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext


ALLOWED_EXT = {".mp3", ".wav", ".ogg", ".silk", ".amr"}
PAGE_SIZE = 15


@dataclass
class AiriListAllVoicesTool(FunctionTool[AstrAgentContext]):
    """列出当前插件中所有可用的语音名称。"""

    name: str = "airi_list_all_voices"
    description: str = "列出本插件加载的全部语音名称，供 LLM 选择使用。"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        }
    )
    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        if not self.plugin or getattr(self.plugin, "trigger_mode", None) != "llm":
            return "当前未开启 LLM 触发模式，本工具暂不可用。"

        if not self.plugin.voice_map:
            return "当前没有可用语音。"

        names = sorted(self.plugin.voice_map.keys())
        return "当前可用语音名称列表：\n" + "\n".join(names)


@dataclass
class AiriSearchVoicesTool(FunctionTool[AstrAgentContext]):
    """根据关键词筛选语音名称。"""

    name: str = "airi_search_voices"
    description: str = (
        "根据用户给出的关键词，在本插件的语音库中筛选匹配的语音名称。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "用户给出的语音关键词，用于模糊匹配语音名称。",
                }
            },
            "required": ["keyword"],
        }
    )
    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        if not self.plugin or getattr(self.plugin, "trigger_mode", None) != "llm":
            return "当前未开启 LLM 触发模式，本工具暂不可用。"

        if not self.plugin.voice_map:
            return "当前没有可用语音。"

        keyword = (kwargs.get("keyword") or "").strip()
        if not keyword:
            return "请提供要搜索的语音关键词。"

        keyword_lower = keyword.lower()
        matched = [
            name
            for name in self.plugin.voice_map.keys()
            if keyword_lower in name.lower()
        ]

        if not matched:
            return f"未找到包含「{keyword}」的语音名称。"

        matched.sort()
        return (
            f"根据关键词「{keyword}」筛选到的语音名称：\n" + "\n".join(matched)
        )


@dataclass
class AiriSendVoiceTool(FunctionTool[AstrAgentContext]):
    """根据指定名称直接向当前会话发送语音。"""

    name: str = "airi_send_voice"
    description: str = (
        "根据指定的语音名称，直接向当前会话发送对应的语音消息。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要发送的语音名称，必须是已存在的语音列表中的一个。",
                }
            },
            "required": ["name"],
        }
    )
    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        if not self.plugin or getattr(self.plugin, "trigger_mode", None) != "llm":
            return "当前未开启 LLM 触发模式，本工具暂不可用。"

        if not self.plugin.voice_map:
            return "当前没有可用语音。"

        name = (kwargs.get("name") or "").strip()
        if not name:
            return "请提供要发送的语音名称。"

        path = self.plugin.voice_map.get(name)
        if not path:
            return f"语音「{name}」不存在，请先使用列出/搜索工具确认可用名称。"

        try:
            agent_ctx = context.context.context
            event = context.context.event
        except Exception:
            agent_ctx = None
            event = None

        if agent_ctx is None or event is None:
            return f"无法获取当前会话上下文，未能发送语音「{name}」。"

        try:
            await agent_ctx.send_message(
                event.unified_msg_origin,
                MessageChain([Record.fromFileSystem(path)]),
            )
            logger.debug(f"[AiriVoice] LLM 工具发送语音：'{name}' → {path}")
            return ""
        except FileNotFoundError as e:
            logger.error(f"[AiriVoice] 文件不存在（LLM 工具） '{name}': {e}")
            return f"语音文件不存在：{name}"
        except Exception as e:
            logger.error(f"[AiriVoice] LLM 工具发送失败 '{name}': {e}")
            return f"语音发送失败：{type(e).__name__}"


@register(
    "airi_voice",
    "lidure",
    "输入关键词发送对应语音 + 口癖自动补语音",
    "2.3",  # 建议小版本+1 表示有新功能
    "https://github.com/Lidure/astrbot_plugin_airi_voice",
)
class AiriVoice(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        
        # 路径初始化
        self.plugin_dir = Path(__file__).parent
        self.voice_dir = self.plugin_dir / "voices"
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = StarTools.get_data_dir("astrbot_plugin_airi_voice")
        
        self.user_added_dir = self.data_dir / "user_added"
        self.user_added_dir.mkdir(parents=True, exist_ok=True)
        
        self.extra_voice_dir = self.data_dir / "extra_voices"
        self.extra_voice_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[AiriVoice] 数据目录：{self.data_dir}")
        
        # 配置
        self.config = config or {}
        self.trigger_mode = self.config.get("trigger_mode", "direct")
        if self.trigger_mode not in {"prefix", "direct", "llm"}:
            self.trigger_mode = "direct"
        
        self.admin_mode = self.config.get("admin_mode", "whitelist")
        if self.admin_mode not in {"all", "admin", "whitelist"}:
            self.admin_mode = "whitelist"
        
        whitelist_raw = self.config.get("admin_whitelist", "")
        if isinstance(whitelist_raw, str):
            self.admin_whitelist: Set[str] = set(
                line.strip() for line in whitelist_raw.splitlines() if line.strip()
            )
        elif isinstance(whitelist_raw, list):
            self.admin_whitelist: Set[str] = set(str(x).strip() for x in whitelist_raw if str(x).strip())
        else:
            self.admin_whitelist: Set[str] = set()

        self.llm_select_mode = self.config.get("llm_select_mode", "list")
        if self.llm_select_mode not in {"list", "keyword"}:
            self.llm_select_mode = "list"
        
        # 新增：口癖自动补语音配置
        catch_config = self.config.get("auto_catchphrase", {})
        self.auto_catchphrase_enabled = catch_config.get("enabled", False)
        self.catchphrase_mode = catch_config.get("mode", "first")  # first / random / all
        self.catchphrase_map: Dict[str, List[str]] = catch_config.get("map", {})
        self.catchphrase_max_per_msg = catch_config.get("max_per_message", 2)

        logger.info(f"[AiriVoice] 口癖自动补语音：{'启用' if self.auto_catchphrase_enabled else '关闭'}，"
                    f"关键词数：{len(self.catchphrase_map)}")

        # 语音映射
        self.voice_map: Dict[str, str] = {}
        self.sorted_keys: List[str] = []
        
        self._load_local_voices()
        self._load_user_added_voices()
        self._load_web_voices(self.config)
        self._update_sorted_keys()
        
        self.last_pool_len = len(self.config.get("extra_voice_pool", []))
        
        if self.trigger_mode == "llm":
            llm_tools = []
            if self.llm_select_mode == "list":
                llm_tools.append(AiriListAllVoicesTool(plugin=self))
            else:
                llm_tools.append(AiriSearchVoicesTool(plugin=self))
            llm_tools.append(AiriSendVoiceTool(plugin=self))

            try:
                self.context.add_llm_tools(*llm_tools)
                logger.info(
                    f"[AiriVoice] 已为 LLM 注册 {len(llm_tools)} 个语音工具，模式：{self.llm_select_mode}"
                )
            except Exception as e:
                logger.error(f"[AiriVoice] 注册 LLM 工具失败：{e}")
        
        logger.info(f"[AiriVoice] 初始化完成，共 {len(self.voice_map)} 个语音")

    # ... (保持原有的 _get_user_id, _get_reply_id, _get_audio_url, _download_audio, _get_file_ext_from_url 等方法不变)

    def _update_sorted_keys(self):
        self.sorted_keys = sorted(self.voice_map.keys())

    # ... (保持原有的 _load_local_voices, _load_user_added_voices, _load_web_voices, _check_admin 方法不变)

    def _try_auto_send_catchphrase(self, event: AstrMessageEvent, text: str):
        """检查文本中的口癖关键词，并尝试补发语音"""
        if not self.auto_catchphrase_enabled or not text:
            return

        matched_voices = []

        for keyword, voice_list in self.catchphrase_map.items():
            if keyword in text:
                if self.catchphrase_mode == "all":
                    matched_voices.extend(voice_list)
                elif self.catchphrase_mode == "random":
                    if voice_list:
                        matched_voices.append(random.choice(voice_list))
                else:  # first
                    if voice_list:
                        matched_voices.append(voice_list[0])

        if not matched_voices:
            return

        # 限制数量，避免刷屏
        matched_voices = matched_voices[:self.catchphrase_max_per_msg]
        random.shuffle(matched_voices)  # 美观

        for vname in matched_voices:
            path = self.voice_map.get(vname)
            if path and Path(path).exists():
                try:
                    yield event.chain_result([Record.fromFileSystem(path)])
                    logger.info(f"[AiriVoice auto] 补发语音：'{vname}' (因关键词 '{keyword}')")
                except Exception as e:
                    logger.error(f"[AiriVoice auto] 补发失败 '{vname}': {e}")

    @filter.regex(r"^\s*.+\s*$")
    async def voice_handler(self, event: AstrMessageEvent):
        text = (event.message_str or "").strip()
        if not text:
            return

        # 自动检测配置变化
        current_pool_len = len(self.config.get("extra_voice_pool", []))
        if current_pool_len > self.last_pool_len:
            logger.info("[AiriVoice] 检测到网页配置变化，自动刷新语音列表")
            self._load_web_voices(self.config)
            self._update_sorted_keys()
            self.last_pool_len = current_pool_len

        # 随机语音逻辑（保持原样）
        if text.startswith("随机") and self.voice_map:
            # ... (原随机逻辑不变，省略以节省空间)
            pass  # 请保留你原来的随机处理代码

        # 原关键词发送逻辑
        keyword = text
        if self.trigger_mode == "prefix":
            match = re.search(r"^#voice\s+(.+)", text, re.I)
            if not match:
                return
            keyword = match.group(1).strip()

        matched_path = self.voice_map.get(keyword)
        if matched_path:
            try:
                yield event.chain_result([Record.fromFileSystem(matched_path)])
                logger.debug(f"[AiriVoice] 发送语音：'{keyword}'")
            except Exception as e:
                logger.error(f"[AiriVoice] 发送失败 '{keyword}': {e}")
                yield event.plain_result(f"语音发送失败：{type(e).__name__}")

            # 新增：如果插件自己发送了语音，也检查一下文本（虽然通常没有文本，但以防）
            # 但更重要的是下面 send_text 命令

        # 新增：如果这条消息是插件自己发的文本（极少见），尝试补语音
        # 但实际中更推荐用下面的 /voice.send_text
        if self.auto_catchphrase_enabled:
            for part in event.get_messages():
                if isinstance(part, Plain):
                    yield from self._try_auto_send_catchphrase(event, part.text)

    @filter.command("voice.send_text")
    async def send_text_with_auto_voice(self, event: AstrMessageEvent):
        """
        发送一段文本，并自动检查口癖补发语音
        用法：/voice.send_text 你好啊，打卡啦摩托～
        （管理员命令，适合测试或让 LLM 调用）
        """
        if not self._check_admin(event):
            yield event.plain_result("❌ 此命令仅限管理员使用")
            return

        text = (event.message_str or "").replace("/voice.send_text", "", 1).strip()
        if not text:
            yield event.plain_result("请在命令后输入要发送的文本")
            return

        # 先发送文本
        try:
            yield event.chain_result([Plain(text)])
        except Exception as e:
            yield event.plain_result(f"发送文本失败：{str(e)}")
            return

        # 再检查补语音
        yield from self._try_auto_send_catchphrase(event, text)

    # ... (其余命令如 voice.add, voice.delete, voice.list, voice.reload, voice.help, voice.check 保持不变)

    @filter.command("voice.help")
    async def help(self, event: AstrMessageEvent):
        is_admin = self._check_admin(event)
        
        commands = [
            "📋 /voice.list [页码] - 查看可用语音",
            "❓ /voice.help - 显示此帮助",
        ]
        if is_admin:
            commands += [
                "➕ /voice.add 名字 - 引用语音消息添加新语音",
                "🗑️ /voice.delete 名字 - 删除语音",
                "🔄 /voice.reload - 重新加载语音列表",
                "💬 /voice.send_text 内容 - 发送文本并自动补口癖语音（测试用）",
            ]
        
        help_msg = f"""🌸 AiriVoice 语音插件 v2.3

【使用方法】
直接输入语音关键词即可发送（direct 模式）
或 #voice 关键词（prefix 模式）

【新功能：口癖自动补语音】
在配置中设置 auto_catchphrase 后，
当 bot 发送的文本包含关键词时，会额外补发对应语音。
LLM 模式推荐：在角色设定中教模型使用 [voice:语音名] 标记。

【命令】
{chr(10).join(commands)}"""
        
        yield event.plain_result(help_msg)