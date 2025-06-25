import json
import logging
import warnings
import inspect

from packaging.version import Version
import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.config import AutoLoggingConfig
from mlflow.utils.autologging_utils.safety import safe_patch
from mlflow.utils.annotations import experimental

# Define the flavor name for the Vinagent integration
FLAVOR_NAME = "vinagent"

_logger = logging.getLogger(__name__)

@experimental(api_or_type="3.1.0")
@autologging_integration(FLAVOR_NAME)
def autolog(
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    log_traces=True,
):
    """
    Enables (or disables) and configures autologging from Vinagent to MLflow.

    Args:
        disable: If True, disables the Vinagent autologging integration.
        exclusive: If True, autologged content is not logged to user-created fluent runs.
        disable_for_unsupported_versions: If True, disables autologging for untested versions.
        silent: If True, suppresses all MLflow event logs and warnings.
        log_traces: If True, traces are logged for Vinagent Agent invoke calls.
    """
    if disable:
        return

    try:
        from vinagent.agent.agent import Agent

        safe_patch(
            FLAVOR_NAME,
            Agent,
            "invoke",
            _patched_agent_invoke,
        )
    except ImportError as e:
        _logger.warning(f"Failed to import Agent class: {e}")
    except Exception as e:
        _logger.warning(f"Failed to enable tracing for Vinagent: {e}")

def _patched_agent_invoke(original, self, *args, **kwargs):
    """
    Patches the Agent.invoke method to enable MLflow tracing.
    """
    config = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not config.log_traces:
        return original(self, *args, **kwargs)

    # Construct span name
    fullname = f"{self.__class__.__name__}.invoke"
    # Start an MLflow span
    with mlflow.start_span(name=fullname, span_type=SpanType.AGENT) as span:
        # Construct inputs
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)

        # Set agent-specific attributes
        _set_span_attributes(span, self)

        try:
            # Call the original invoke method
            result = original(self, *args, **kwargs)

            # Handle outputs
            outputs = _process_outputs(result)
            span.set_outputs(outputs)

            # If the result is from an LLM call, set chat attributes
            if inputs.get("query") and hasattr(result, "content"):
                _set_chat_attributes(span, inputs.get("query"), result.content)

            return result
        
        except Exception as e:
            _logger.error(f"Error during Agent.invoke tracing: {e}")
            span.set_status("ERROR", str(e))
            raise

def _construct_full_inputs(func, self, *args, **kwargs):
    """
    Constructs a dictionary of serializable inputs for the invoke method.
    """
    try:
        signature = inspect.signature(func)
        arguments = signature.bind(self, *args, **kwargs).arguments
        arguments.pop("self", None)  # Remove 'self'

        return {
            k: v.__dict__ if hasattr(v, "__dict__") and _is_serializable(v.__dict__) else v
            for k, v in arguments.items()
            if v is not None and _is_serializable(v)
        }
    except Exception as e:
        _logger.warning(f"Failed to construct inputs: {e}")
        return {}

def _is_serializable(value):
    """
    Checks if a value is JSON-serializable.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            json.dumps(value, cls=TraceJSONEncoder, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False

def _set_span_attributes(span: LiveSpan, instance):
    """
    Sets agent-specific attributes on the span.
    """
    try:
        attributes = {
            "description": getattr(instance, "description", None),
            "skills": getattr(instance, "skills", None),
            "user_id": getattr(instance, "_user_id", None),
            "has_compiled_graph": hasattr(instance, "compiled_graph"),
            "has_memory": bool(getattr(instance, "memory", None)),
        }
        # Filter out None values and non-serializable attributes
        for key, value in attributes.items():
            if value is not None and _is_serializable(value):
                span.set_attribute(key, str(value) if isinstance(value, list) else value)
    except Exception as e:
        _logger.warning(f"Failed to set span attributes: {e}")

def _process_outputs(result):
    """
    Processes the output for logging to the span.
    """
    try:
        if result is None:
            return None
        elif hasattr(result, "__dict__") and _is_serializable(result.__dict__):
            return result.__dict__
        elif hasattr(result, "content") and _is_serializable(result.content):
            return {"content": result.content}
        elif _is_serializable(result):
            return result
        else:
            return str(result)
    except Exception as e:
        _logger.warning(f"Failed to process outputs: {e}")
        return str(result)

def _set_chat_attributes(span: LiveSpan, query: str, output: str):
    """
    Sets chat-specific attributes for LLM interactions.
    """
    try:
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": output},
        ]
        span.set_attribute("chat_messages", messages)
    except Exception as e:
        _logger.warning(f"Failed to set chat attributes: {e}")
