import inspect
from abc import ABC, abstractmethod
from typing import Union, Dict, Optional, Any
from langchain_core.runnables import RunnableConfig
from vinagent.graph.node import Node
from vinagent.agent import Agent


class AgentNode(Agent, Node):
    def __init__(self, name: Optional[str] = None, config: Any = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name or self.__class__.__name__
        self.config = config
        self._current_user_id = "unknown_user"
        self._current_thread_id = 123

    def __call__(self, state: Any, config: RunnableConfig = None) -> dict:
        return self.exec(state, config)

    def with_config(self, config):
        self.config = config

    def invoke(self, prompt: str, max_iterations: int = 5, **kwargs):
        # Automatically forward user_id into every self.invoke() call
        if self.config:
            if "configurable" in self.config:
                if "user_id" in self.config["configurable"]:
                    self._current_user_id = self.config["configurable"]["user_id"]
                else:
                    raise ValueError(
                        """config must be {'configurable': {'user_id': 'unknown_user'}, 'thread_id': 123"}"""
                    )
            if "thread_id" in self.config:
                self._current_thread_id = self.config["thread_id"]
            else:
                raise ValueError(
                    """config must be {'configurable': {'user_id': 'unknown_user'}, 'thread_id': 123"}"""
                )
        return super().invoke(
            prompt,
            max_iterations=max_iterations,
            user_id=self._current_user_id,
            thread_id=self._current_thread_id,
            **kwargs,
        )

    def branching(self, state: Any, config: Optional[RunnableConfig] = None) -> str:
        pass

    def __rshift__(self, other: Union["Node", Dict[str, "Node"], str]) -> "Node":
        self.target = other
        return self


class UserFeedback(ABC, Node):
    def __init__(
        self, name: str = "user_feedback", role: str = "user", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.role = role

    @abstractmethod
    def exec(
        self, state: Optional[Any], config: Optional[RunnableConfig] = None
    ) -> Union[dict, str]:
        raise NotImplementedError("Subclasses must implement exec method")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        child_sig = inspect.signature(cls.exec)

        # Compare parameter details
        child_params = [p.name for p in child_sig.parameters.values()]

        if "state" not in child_params:
            print(child_params)
            raise TypeError(
                f"Your are missing 'state' argument in exec() method. Fix by adding like: exec(self, state: State)"
            )
