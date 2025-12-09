import logging
import base64
import json
from typing import Dict, List, Tuple
import psutil
import time

import requests

from mm_agents.prompts import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE,
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE,
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_SOM_OUT_TAG,
    SYS_PROMPT_LIBREOFFICE_CALC,
    SYS_PROMPT_SIMPLIFIED
)
from mm_agents.agent import (
    encode_image,
    linearize_accessibility_tree,
    trim_accessibility_tree,
    parse_actions_from_string,
    parse_code_from_string,
    parse_code_from_som_string,
)

logger = logging.getLogger("desktopenv.agent")

def get_mem_mb():
    return psutil.Process(95238).memory_info().rss / 1024**2

class MyAgent:
    def __init__(
        self,
        platform: str = "ubuntu",
        model: str = "dummy",
        max_tokens: int = 0,
        top_p: float = 0.0,
        temperature: float = 0.0,
        action_space: str = "pyautogui",
        observation_type: str = "a11y_tree",
        max_trajectory_length: int = 0,
        a11y_tree_max_tokens: int = 10000,
        client_password: str = "password",
    ):
        # mirror PromptAgent's constructor signature and public attributes
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.client_password = client_password

        self.thoughts: List[str] = []
        self.actions: List[List] = []
        self.observations: List[Dict] = []

        self.system_message=SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION

        # self.system_message = self.system_message.format(CLIENT_PASSWORD=self.client_password)
        # self.system_message = SYS_PROMPT_SIMPLIFIED

    def _format_last_actions(self) -> str:
        """Return a concise string describing the previous step's actions, if any."""
        if not self.actions:
            return "No previous action was taken."
        last = self.actions[-1]
        if not last:
            return "No previous action was taken."
        try:
            if isinstance(last, list):
                return "Previous action(s): " + "; ".join(map(str, last))
            return f"Previous action(s): {last}"
        except Exception:
            return "Previous action(s): (unavailable)"

    def _build_messages(self, instruction: str, obs: Dict) -> List[Dict]:
        system_message = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)
        messages: List[Dict] = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_message},
                ],
            }
        ]

        # Prepare previous action summary to include in the prompt
        prev_action_summary = self._format_last_actions()

        # Only append current observation (we keep history minimal for local calls)
        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            base64_image = encode_image(obs["screenshot"]) if obs.get("screenshot") else None
            linearized_accessibility_tree = (
                linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"], platform=self.platform)
                if self.observation_type == "screenshot_a11y_tree" and obs.get("accessibility_tree")
                else None
            )
            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )
            # Track
            self.observations.append({
                "screenshot": base64_image,
                "accessibility_tree": linearized_accessibility_tree,
            })
            # Compose message
            if self.observation_type == "screenshot":
                text_prompt = (
                    "Given the screenshot as below. What's the next step that you will do to help with the task?\n"
                    # + prev_action_summary
                )
            else:
                text_prompt = (
                    "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                        linearized_accessibility_tree or ""
                    )
                    + "\n" + prev_action_summary
                )
            content_parts = [{"type": "text", "text": text_prompt}]
            if base64_image:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"},
                })
            messages.append({"role": "user", "content": content_parts})

        elif self.observation_type == "a11y_tree":
            linearized_accessibility_tree = linearize_accessibility_tree(
                accessibility_tree=obs["accessibility_tree"], platform=self.platform
            )
            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )
            self.observations.append({"screenshot": None, "accessibility_tree": linearized_accessibility_tree})
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?\n{}".format(
                            linearized_accessibility_tree,
                            prev_action_summary,
                        ),
                    }
                ],
            })
        elif self.observation_type == "som":
            # Not implementing SOM tagging flow here for simplicity
            raise ValueError("'som' observation_type is not supported by MyAgent yet")
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type)

        return messages

    def _call_local_llm(self, messages: List[Dict]) -> str:
        url = "http://localhost:8080/v1/chat/completions"
        model_name = self.model or "Qwen3-VL-2B-Instruct-GGUF:Q4_K_M"
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max(self.max_tokens, 64) if self.max_tokens else 256,
            "stream": False,
        }
        logger.info("MyAgent calling local LLM model=%s", model_name)
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content

    def _parse_actions(self, response: str) -> List:
        # Mirror PromptAgent.parse_actions minimal logic
        if self.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            if self.action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + self.action_space)
            self.actions.append(actions)
            return actions
        elif self.observation_type == "som":
            if self.action_space == "pyautogui":
                # Not supported (we didn't create masks here)
                raise ValueError("SOM parsing not supported in MyAgent")
            else:
                raise ValueError("Invalid action space: " + self.action_space)
        else:
            raise ValueError("Invalid observation type: " + self.observation_type)

    def predict(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        """
        Predict next action(s) using local llama.cpp server (OpenAI-compatible).
        Use project system prompts and parsing utilities to keep behavior aligned with PromptAgent.
        """
        mem_before=get_mem_mb()
        response_meta = {"instruction": instruction}
        try:
            messages = self._build_messages(instruction, obs)
            content = self._call_local_llm(messages)
            actions = self._parse_actions(content)
            thought = content
        except Exception as e:
            logger.warning("MyAgent local LLM error, fallback to right-click: %s", e)
            actions = ["pyautogui.click(button='right')"] if self.action_space == "pyautogui" else ["WAIT"]
            thought = "Fallback due to LLM error"

        mem_after=get_mem_mb()
        with open('mem_log.txt', 'a') as f:
            f.write(f'{time.time()} -- Memory before: {mem_before} MB, after: {mem_after} MB, consume: {mem_after-mem_before} MB\n')
        response_meta["thought"] = thought
        logger.info("MyAgent returns actions: %s", actions)
        return response_meta, actions

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent")
        self.thoughts = []
        self.actions = []
        self.observations = []