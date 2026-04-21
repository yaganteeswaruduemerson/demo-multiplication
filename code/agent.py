import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import re
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from config import Config
import openai
import json

# =========================
# CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are a professional customer service technical advisor specializing in Python programming. "
    "Your task is to generate a comprehensive specification for multiplying two numbers using Python. "
    "Provide clear instructions, sample code, and explanations suitable for customer service representatives and clients. "
    "Ensure your response is formal, accurate, and easy to follow. If the input is invalid or outside the scope of multiplication, "
    "explain the issue and suggest corrective actions. Format your output with code blocks and step-by-step guidance. "
    "If information is not available, politely inform the user and offer to escalate to technical support."
)
OUTPUT_FORMAT = "Step-by-step instructions, Python code block, and explanatory notes."
FALLBACK_RESPONSE = (
    "I'm unable to generate the specification for multiplication at this time. "
    "Please ensure your request is correctly formatted or contact technical support for further assistance."
)
FEW_SHOT_EXAMPLES = [
    "Multiply 3 and 5 using Python.\nTo multiply 3 and 5 in Python, use the following code:\n\n```python\nresult = 3 * 5\nprint(result)  # Output: 15\n```\nThis code multiplies the two numbers and prints the result.",
    "Multiply 7 and 'abc' using Python.\nThe input 'abc' is not numeric. Please provide two valid numbers for multiplication."
]
USER_PROMPT_TEMPLATE = "Please provide two numbers to multiply using Python. Ensure your input is numeric."

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(__import__('pathlib').Path(__file__).parent / "validation_config.json")

# =========================
# LOGGING CONFIG
# =========================

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# =========================
# INPUT/OUTPUT MODELS
# =========================

class MultiplicationRequest(BaseModel):
    number1: Any = Field(..., description="First number to multiply")
    number2: Any = Field(..., description="Second number to multiply")
    operation_type: Optional[str] = Field("multiplication", description="Operation type (must be 'multiplication')")

    @field_validator("number1", "number2")
    @classmethod
    def validate_number(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                raise ValueError("Input cannot be empty.")
            try:
                # Accept int or float
                if "." in v:
                    return float(v)
                return int(v)
            except Exception:
                raise ValueError("Input must be numeric.")
        if isinstance(v, (int, float)):
            return v
        raise ValueError("Input must be numeric.")

    @field_validator("operation_type")
    @classmethod
    def validate_operation_type(cls, v):
        if v is None:
            return "multiplication"
        v = str(v).strip().lower()
        if v != "multiplication":
            raise ValueError("Only 'multiplication' operation is supported.")
        return v

    @model_validator(mode="after")
    def validate_both_numbers(self):
        if self.number1 is None or self.number2 is None:
            raise ValueError("Both numbers must be provided.")
        return self

class MultiplicationResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[str] = Field(None, description="Formatted specification/code/explanation")
    error: Optional[str] = Field(None, description="Error message if any")
    error_code: Optional[str] = Field(None, description="Error code if any")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input errors")

# =========================
# SANITIZER UTILITY
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# SERVICE CLASSES
# =========================

class InputHandler:
    """Receives and preprocesses user input."""

    def receive_input(self, user_query: str) -> Dict[str, Any]:
        """
        Extracts numbers and operation type from user query.
        Returns:
            dict with keys: number1, number2, operation_type
        Raises:
            ValueError if parsing fails.
        """
        # Accepts queries like "Multiply 3 and 5", "3 * 5", "3,5", etc.
        try:
            # Try to extract two numbers from the string
            # Accepts: "Multiply 3 and 5", "3 * 5", "3,5", "3 and 5"
            pattern = r"(-?\d+(?:\.\d+)?)\s*(?:\*|and|,|x|\s)\s*(-?\d+(?:\.\d+)?)"
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                number1 = match.group(1)
                number2 = match.group(2)
                return {
                    "number1": number1,
                    "number2": number2,
                    "operation_type": "multiplication"
                }
            # Try to find numbers in the string
            numbers = re.findall(r"-?\d+(?:\.\d+)?", user_query)
            if len(numbers) >= 2:
                return {
                    "number1": numbers[0],
                    "number2": numbers[1],
                    "operation_type": "multiplication"
                }
            # If the user says "multiply X by Y"
            pattern2 = r"multiply\s+(-?\d+(?:\.\d+)?)\s+(?:by|and)\s+(-?\d+(?:\.\d+)?)"
            match2 = re.search(pattern2, user_query, re.IGNORECASE)
            if match2:
                number1 = match2.group(1)
                number2 = match2.group(2)
                return {
                    "number1": number1,
                    "number2": number2,
                    "operation_type": "multiplication"
                }
            raise ValueError("Could not extract two numbers for multiplication.")
        except Exception as e:
            logger.error(f"InputHandler.receive_input error: {e}")
            raise ValueError("INVALID_INPUT")

class InputValidator:
    """Validates that both inputs are numeric and operation is multiplication."""

    def validate(self, number1: Any, number2: Any, operation_type: Optional[str]) -> Dict[str, Any]:
        """
        Validates numeric input and operation type.
        Returns:
            dict with validated numbers and operation_type
        Raises:
            ValueError with error code if validation fails.
        """
        try:
            # Use Pydantic model for validation
            req = MultiplicationRequest(number1=number1, number2=number2, operation_type=operation_type)
            return {
                "number1": req.number1,
                "number2": req.number2,
                "operation_type": req.operation_type
            }
        except ValidationError as ve:
            logger.error(f"InputValidator.validate error: {ve}")
            raise ValueError("INVALID_INPUT")
        except Exception as e:
            logger.error(f"InputValidator.validate error: {e}")
            raise ValueError("INVALID_INPUT")

class LLMService:
    """Handles interaction with Azure OpenAI GPT-4.1."""

    def __init__(self):
        self._client = None

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_client(self):
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not configured")
        if self._client is None:
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
        few_shot_examples: Optional[List[str]] = None
    ) -> str:
        """
        Calls Azure OpenAI GPT-4.1 with constructed prompt.
        Returns:
            str (LLM response)
        Raises:
            Exception on API error.
        """
        messages = []
        # System prompt with output format
        sys_msg = system_prompt.strip() + "\n\nOutput Format: " + OUTPUT_FORMAT
        messages.append({"role": "system", "content": sys_msg})
        # Few-shot examples as user/assistant pairs
        if few_shot_examples:
            for example in few_shot_examples:
                # Heuristic: If example contains a code block, treat as user/assistant pair
                if "\n```" in example:
                    parts = example.split("\n", 1)
                    user_example = parts[0].strip()
                    assistant_example = parts[1].strip() if len(parts) > 1 else ""
                    messages.append({"role": "user", "content": user_example})
                    if assistant_example:
                        messages.append({"role": "assistant", "content": assistant_example})
                else:
                    messages.append({"role": "user", "content": example})
        # User prompt
        messages.append({"role": "user", "content": user_prompt})
        # Optionally add context (not used here)
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        client = self.get_llm_client()
        try:
            response = await client.chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4.1",
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return content
        except Exception as e:
            logger.error(f"LLMService.call_llm error: {e}")
            raise

class SpecificationGenerator:
    """Generates Python multiplication specification, code samples, and explanations."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def generate_specification(self, validated_numbers: Dict[str, Any]) -> str:
        """
        Generates multiplication specification using LLM.
        Returns:
            str (LLM output)
        Raises:
            Exception on LLM error.
        """
        number1 = validated_numbers["number1"]
        number2 = validated_numbers["number2"]
        operation_type = validated_numbers.get("operation_type", "multiplication")
        # Compose user prompt
        user_prompt = f"Multiply {number1} and {number2} using Python."
        # Call LLM
        llm_output = await self.llm_service.call_llm(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            context=None,
            few_shot_examples=FEW_SHOT_EXAMPLES
        )
        return llm_output

class ResponseFormatter:
    """Formats the response according to output instructions."""

    def format_response(self, llm_output: str) -> str:
        """
        Formats LLM output as per output instructions.
        Returns:
            str (formatted response)
        """
        try:
            # Sanitize LLM output
            formatted = sanitize_llm_output(llm_output, content_type="code")
            return formatted
        except Exception as e:
            logger.error(f"ResponseFormatter.format_response error: {e}")
            return FALLBACK_RESPONSE

class ErrorHandler:
    """Handles errors, retries, fallback responses, and escalation."""

    def handle_error(self, error_code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handles errors, provides fallback or escalation.
        Returns:
            dict with error response
        """
        logger.error(f"ErrorHandler.handle_error: {error_code}, context={context}")
        if error_code == "INVALID_INPUT":
            return {
                "success": False,
                "result": None,
                "error": "Invalid input. Please provide two valid numbers for multiplication.",
                "error_code": "INVALID_INPUT",
                "tips": "Ensure both inputs are numeric values (e.g., 3 and 5)."
            }
        elif error_code == "UNSUPPORTED_OPERATION":
            return {
                "success": False,
                "result": None,
                "error": "Only multiplication operation is supported.",
                "error_code": "UNSUPPORTED_OPERATION",
                "tips": "Please specify multiplication only."
            }
        elif error_code == "API_ERROR":
            return {
                "success": False,
                "result": None,
                "error": "There was a problem communicating with the language model.",
                "error_code": "API_ERROR",
                "tips": "Please try again later or contact technical support."
            }
        else:
            return {
                "success": False,
                "result": None,
                "error": FALLBACK_RESPONSE,
                "error_code": error_code,
                "tips": "Check your input and try again."
            }

# =========================
# MAIN AGENT CLASS
# =========================

class MultiplicationAgent:
    """
    Orchestrates the flow between components, manages session state, and delivers final response.
    """

    def __init__(self):
        self.input_handler = InputHandler()
        self.input_validator = InputValidator()
        self.llm_service = LLMService()
        self.specification_generator = SpecificationGenerator(self.llm_service)
        self.response_formatter = ResponseFormatter()
        self.error_handler = ErrorHandler()
        self.guardrails_config = GUARDRAILS_CONFIG

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main entry point; orchestrates input handling, validation, specification generation, and response formatting.
        Returns:
            dict (MultiplicationResponse)
        """
        async with trace_step(
            "input_processing",
            step_type="parse",
            decision_summary="Extract and validate numbers and operation from user query",
            output_fn=lambda r: f"input={user_query[:50]}"
        ) as step:
            try:
                parsed_input = self.input_handler.receive_input(user_query)
                step.capture(parsed_input)
            except Exception as e:
                step.capture({"error": str(e)})
                return self.error_handler.handle_error("INVALID_INPUT", {"user_query": user_query})

        async with trace_step(
            "input_validation",
            step_type="parse",
            decision_summary="Validate numeric input and operation type",
            output_fn=lambda r: f"validated={r}"
        ) as step:
            try:
                validated = self.input_validator.validate(
                    parsed_input["number1"],
                    parsed_input["number2"],
                    parsed_input.get("operation_type", "multiplication")
                )
                step.capture(validated)
            except Exception as e:
                step.capture({"error": str(e)})
                return self.error_handler.handle_error("INVALID_INPUT", parsed_input)

        async with trace_step(
            "llm_specification_generation",
            step_type="llm_call",
            decision_summary="Generate multiplication specification using LLM",
            output_fn=lambda r: f"llm_output={str(r)[:50]}"
        ) as step:
            try:
                llm_output = await self.specification_generator.generate_specification(validated)
                step.capture(llm_output)
            except Exception as e:
                step.capture({"error": str(e)})
                return self.error_handler.handle_error("API_ERROR", validated)

        async with trace_step(
            "response_formatting",
            step_type="format",
            decision_summary="Format LLM output for user",
            output_fn=lambda r: f"formatted={str(r)[:50]}"
        ) as step:
            try:
                formatted = self.response_formatter.format_response(llm_output)
                step.capture(formatted)
            except Exception as e:
                step.capture({"error": str(e)})
                return self.error_handler.handle_error("FORMAT_ERROR", {"llm_output": llm_output})

        return {
            "success": True,
            "result": formatted,
            "error": None,
            "error_code": None,
            "tips": None
        }

# =========================
# OBSERVABILITY LIFESPAN
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Customer Service Multiplication Assistant",
    description="Professional customer service agent for generating Python multiplication specifications, code, and explanations.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/multiply", response_model=MultiplicationResponse)
async def multiply_endpoint(req: MultiplicationRequest):
    """
    Multiply two numbers using Python and return a specification, code, and explanation.
    """
    agent = MultiplicationAgent()
    try:
        async with trace_step(
            "api_entry",
            step_type="parse",
            decision_summary="API entrypoint for multiplication",
            output_fn=lambda r: f"req={req}"
        ) as step:
            # Compose user query string for InputHandler
            user_query = f"Multiply {req.number1} and {req.number2} using Python."
            result = await agent.process_query(user_query)
            step.capture(result)
            # Sanitize LLM output before returning
            if result.get("result"):
                result["result"] = sanitize_llm_output(result["result"], content_type="code")
            return result
    except Exception as e:
        logger.error(f"multiply_endpoint error: {e}")
        return {
            "success": False,
            "result": None,
            "error": FALLBACK_RESPONSE,
            "error_code": "SYSTEM_ERROR",
            "tips": "Please try again later or contact technical support."
        }

# =========================
# ERROR HANDLING
# =========================

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "result": None,
            "error": "Invalid input format.",
            "error_code": "INVALID_INPUT",
            "tips": "Check your input values. Ensure both numbers are provided and numeric."
        }
    )

@app.exception_handler(json.decoder.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.decoder.JSONDecodeError):
    logger.error(f"Malformed JSON: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "result": None,
            "error": "Malformed JSON request.",
            "error_code": "MALFORMED_JSON",
            "tips": "Ensure your JSON is properly formatted (quotes, commas, etc.)."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "result": None,
            "error": FALLBACK_RESPONSE,
            "error_code": "SYSTEM_ERROR",
            "tips": "Please try again later or contact technical support."
        }
    )

# =========================
# MAIN ENTRYPOINT
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())