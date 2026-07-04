#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后端适配模块

职责：
  - 默认使用 vLLM (OpenAI 兼容) 后端，行为与原流程完全一致
  - 可选启用自定义后端 (custom_backend.enabled=true)
  - 请求/响应格式集中封装在 BackendClient 子类中，不散落在 process.py
  - 支持从外部文件加载 prompt，不覆盖默认值
"""

import os
import requests
from openai import OpenAI


# =============================================================================
# BackendClient 基类
# =============================================================================

class BackendClient:
    """统一的聊天补全接口。所有后端实现必须继承此类。"""

    def chat_completion(self, messages: list, inference_config: dict) -> str:
        """发送聊天补全请求，返回模型生成的文本内容。"""
        raise NotImplementedError

    def get_model_name(self) -> str:
        """返回当前后端的模型名称标识。"""
        raise NotImplementedError


# =============================================================================
# VLLMBackendClient — 默认 vLLM 后端
# =============================================================================

class VLLMBackendClient(BackendClient):
    """原生 vLLM OpenAI 兼容客户端。

    保持与原 process.py 完全相同的调用逻辑和参数设置。
    当 custom_backend.enabled=false 时使用此客户端。
    """

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = OpenAI(base_url=server_url, api_key="not-needed")
        self._model_name = None

    def get_model_name(self) -> str:
        if self._model_name is None:
            models = self.client.models.list()
            self._model_name = models.data[0].id
            print(f"[INFO] 模型: {self._model_name}")
        return self._model_name

    def chat_completion(self, messages: list, inference_config: dict) -> str:
        thinking_enabled = inference_config.get("thinking_enabled", False)
        extra_body = {
            "top_k": inference_config.get("top_k", 20),
            "min_p": inference_config.get("min_p", 0.0),
            "repetition_penalty": inference_config.get("repetition_penalty", 1.0),
        }
        if not thinking_enabled:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        response = self.client.chat.completions.create(
            model=self.get_model_name(),
            messages=messages,
            temperature=inference_config.get("temperature", 0.0),
            max_tokens=inference_config.get("max_tokens", 4096),
            top_p=inference_config.get("top_p", 0.95),
            presence_penalty=inference_config.get("presence_penalty", 0.0),
            extra_body=extra_body,
            stream=False,
        )

        return response.choices[0].message.content


# =============================================================================
# CustomBackendClient — 自定义后端适配
# =============================================================================

class CustomBackendClient(BackendClient):
    """自定义后端适配客户端。

    请求/响应格式集中在此处封装：
      - _build_request() 负责请求格式转换
      - _parse_response() 负责响应格式提取

    避免适配逻辑散落在 process.py 各处。
    """

    def __init__(self, base_url: str, model_name: str = "custom-backend",
                 timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def get_model_name(self) -> str:
        return self.model_name

    def chat_completion(self, messages: list, inference_config: dict) -> str:
        payload = self._build_request(messages, inference_config)

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"自定义后端请求超时 ({self.timeout}s): {self.base_url}"
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"无法连接自定义后端: {self.base_url}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"自定义后端返回 HTTP {e.response.status_code}: "
                f"{e.response.text[:500]}"
            )

    def _build_request(self, messages: list, inference_config: dict) -> dict:
        """构建请求体。

        如需适配不同的 API 格式，在此方法中集中处理请求映射。
        默认发送 OpenAI 兼容格式。
        """
        return {
            "model": self.model_name,
            "messages": messages,
            "temperature": inference_config.get("temperature", 0.0),
            "max_tokens": inference_config.get("max_tokens", 4096),
            "top_p": inference_config.get("top_p", 0.95),
        }

    def _parse_response(self, data: dict) -> str:
        """解析响应体。

        如需适配不同的响应结构，在此方法中集中处理响应映射。
        默认解析 OpenAI 兼容格式。
        """
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"自定义后端响应格式异常: {e}\n"
                f"期望 data.choices[0].message.content，"
                f"收到: {str(data)[:500]}"
            )


# =============================================================================
# 工厂函数
# =============================================================================

def create_backend(config: dict) -> BackendClient:
    """根据配置创建后端客户端。

    规则：
      - custom_backend.enabled=false（默认）→ VLLMBackendClient
      - custom_backend.enabled=true → CustomBackendClient
      - base_url 同时支持 config.yaml 和环境变量 CUSTOM_BACKEND_URL
    """
    custom = config.get("custom_backend", {})

    if custom.get("enabled", False):
        base_url = (
            custom.get("base_url", "")
            or os.environ.get("CUSTOM_BACKEND_URL", "")
        )
        if not base_url:
            raise ValueError(
                "custom_backend.enabled=true 但未配置 base_url。"
                "请在 config.yaml 的 custom_backend.base_url 中设置后端地址，"
                "或设置环境变量 CUSTOM_BACKEND_URL。"
            )
        model_name = custom.get("model_name", "custom-backend")
        timeout = custom.get("timeout_seconds", 60)
        print(f"[INFO] 使用自定义后端: {base_url} (模型: {model_name})")
        return CustomBackendClient(base_url, model_name, timeout)

    # 默认：vLLM（保持原有行为）
    server_url = os.environ.get(
        "VLLM_SERVER_URL", "http://vllm-server:8000/v1"
    )
    print(f"[INFO] 使用 vLLM 后端: {server_url}")
    return VLLMBackendClient(server_url)


# =============================================================================
# Prompt 文件加载
# =============================================================================

def load_prompt(config: dict, prompt_key: str, default_value: str,
                env_var_name: str = None) -> str:
    """从外部文件加载 prompt，不存在时返回默认值。

    规则：
      - 未配置路径 → 返回 default_value（config.yaml 中的原有 prompt）
      - 配置了路径 → 从文件读取
      - 文件不存在 → 抛出 FileNotFoundError（不静默回退）
      - 环境变量优先级高于 config.yaml 中的路径

    Args:
        config: 完整配置字典
        prompt_key: prompts 段下的键名，如 "ocr_prompt_path"
        default_value: 默认 prompt 字符串
        env_var_name: 可选环境变量名，用于覆盖文件路径

    Returns:
        prompt 文本内容
    """
    prompts_config = config.get("prompts", {})
    file_path = prompts_config.get(prompt_key, "")

    # 环境变量可覆盖文件路径
    if env_var_name:
        env_path = os.environ.get(env_var_name, "")
        if env_path:
            file_path = env_path

    if not file_path:
        return default_value

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Prompt 文件不存在: {file_path}\n"
            f"请检查 config.yaml 中 prompts.{prompt_key} 的路径配置，"
            f"或确保环境变量 {env_var_name or '(未设置)'} 指向有效文件。"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    print(f"[INFO] 从外部文件加载 prompt ({prompt_key}): {file_path}")
    return content
