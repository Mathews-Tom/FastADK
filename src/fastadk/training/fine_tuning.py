"""
Fine-tuning utilities for FastADK.

This module provides helpers for fine-tuning language models with custom data,
making it easier to create specialized models for specific domains.
"""

# mypy: disable-error-code="comparison-overlap,no-any-return"

import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.exceptions import ConfigurationError

logger = logging.getLogger("fastadk.training")


class FineTuningProvider(str, Enum):
    """Supported providers for fine-tuning."""

    OPENAI = "openai"
    VERTEX = "vertex"
    HUGGINGFACE = "huggingface"


class DataFormat(str, Enum):
    """Supported data formats for fine-tuning."""

    OPENAI = "openai"
    VERTEX = "vertex"
    ALPACA = "alpaca"
    JSONL = "jsonl"


class FineTuningConfig:
    """Configuration for fine-tuning a model."""

    def __init__(
        self,
        provider: Union[str, FineTuningProvider],
        base_model: str,
        training_file: str,
        validation_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """
        Initialize fine-tuning configuration.

        Args:
            provider: The provider to use for fine-tuning
            base_model: The base model to fine-tune
            training_file: Path to the training data file
            validation_file: Optional path to the validation data file
            output_dir: Directory to save the fine-tuned model and artifacts
            hyperparameters: Optional hyperparameters for fine-tuning
            suffix: Optional suffix to add to the fine-tuned model name
        """
        self.provider = (
            FineTuningProvider(provider) if isinstance(provider, str) else provider
        )
        self.base_model = base_model
        self.training_file = Path(training_file)
        self.validation_file = Path(validation_file) if validation_file else None
        self.output_dir = (
            Path(output_dir) if output_dir else Path("./fine_tuned_models")
        )
        self.hyperparameters = hyperparameters or {}
        self.suffix = suffix

        # Validate
        if not self.training_file.exists():
            raise ConfigurationError(f"Training file not found: {self.training_file}")

        if self.validation_file and not self.validation_file.exists():
            raise ConfigurationError(
                f"Validation file not found: {self.validation_file}"
            )


class FineTuningJob:
    """Represents a fine-tuning job."""

    def __init__(
        self,
        job_id: str,
        provider: FineTuningProvider,
        status: str,
        created_at: float,
        base_model: str,
        fine_tuned_model: Optional[str] = None,
    ) -> None:
        """
        Initialize a fine-tuning job.

        Args:
            job_id: The ID of the job
            provider: The provider running the job
            status: The current status of the job
            created_at: Timestamp when the job was created
            base_model: The base model being fine-tuned
            fine_tuned_model: The name of the fine-tuned model (if completed)
        """
        self.job_id = job_id
        self.provider = provider
        self.status = status
        self.created_at = created_at
        self.base_model = base_model
        self.fine_tuned_model = fine_tuned_model

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FineTuningJob":
        """
        Create a FineTuningJob from a dictionary.

        Args:
            data: Dictionary with job data

        Returns:
            A FineTuningJob instance
        """
        return cls(
            job_id=data["job_id"],
            provider=FineTuningProvider(data["provider"]),
            status=data["status"],
            created_at=data["created_at"],
            base_model=data["base_model"],
            fine_tuned_model=data.get("fine_tuned_model"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the job to a dictionary.

        Returns:
            Dictionary representation of the job
        """
        return {
            "job_id": self.job_id,
            "provider": self.provider.value,
            "status": self.status,
            "created_at": self.created_at,
            "base_model": self.base_model,
            "fine_tuned_model": self.fine_tuned_model,
        }


class FineTuner:
    """
    Utility for fine-tuning language models.

    This class provides a simplified interface for fine-tuning models with
    different providers like OpenAI, Vertex AI, and Hugging Face.
    """

    def __init__(self) -> None:
        """Initialize the fine-tuner."""
        self._providers: Dict[FineTuningProvider, Dict[str, Any]] = {}

        # Register built-in providers
        self._register_openai()
        self._register_vertex()
        self._register_huggingface()

    def _register_openai(self) -> None:
        """Register OpenAI provider if available."""
        try:
            import openai

            async def openai_create_job(config: FineTuningConfig) -> FineTuningJob:
                # Ensure API key is set
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ConfigurationError(
                        "OPENAI_API_KEY environment variable not set"
                    )

                client = openai.OpenAI(api_key=api_key)

                # Upload files
                training_file_id = None
                validation_file_id = None

                # Upload training file
                with open(config.training_file, "rb") as file:
                    response = client.files.create(
                        file=file,
                        purpose="fine-tune",
                    )
                    training_file_id = response.id

                # Upload validation file if provided
                if config.validation_file:
                    with open(config.validation_file, "rb") as file:
                        response = client.files.create(
                            file=file,
                            purpose="fine-tune",
                        )
                        validation_file_id = response.id

                # Create fine-tuning job
                job_params = {
                    "training_file": training_file_id,
                    "model": config.base_model,
                }

                if validation_file_id:
                    job_params["validation_file"] = validation_file_id

                # Add hyperparameters
                for key, value in config.hyperparameters.items():
                    if key in ["n_epochs", "batch_size", "learning_rate_multiplier"]:
                        job_params[key] = value

                # Create the job
                response = client.fine_tuning.jobs.create(**job_params)  # type: ignore

                # Create job object
                job = FineTuningJob(
                    job_id=response.id,
                    provider=FineTuningProvider.OPENAI,
                    status=response.status,
                    created_at=response.created_at,
                    base_model=config.base_model,
                )

                return job

            async def openai_get_job(job_id: str) -> FineTuningJob:
                # Ensure API key is set
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ConfigurationError(
                        "OPENAI_API_KEY environment variable not set"
                    )

                client = openai.OpenAI(api_key=api_key)

                # Get job status
                response = client.fine_tuning.jobs.retrieve(job_id)

                # Create job object
                job = FineTuningJob(
                    job_id=response.id,
                    provider=FineTuningProvider.OPENAI,
                    status=response.status,
                    created_at=response.created_at,
                    base_model=response.model,
                    fine_tuned_model=response.fine_tuned_model,
                )

                return job

            async def openai_list_jobs() -> List[FineTuningJob]:
                # Ensure API key is set
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ConfigurationError(
                        "OPENAI_API_KEY environment variable not set"
                    )

                client = openai.OpenAI(api_key=api_key)

                # List jobs
                response = client.fine_tuning.jobs.list()

                # Create job objects
                jobs = []
                for job_data in response.data:
                    job = FineTuningJob(
                        job_id=job_data.id,
                        provider=FineTuningProvider.OPENAI,
                        status=job_data.status,
                        created_at=job_data.created_at,
                        base_model=job_data.model,
                        fine_tuned_model=job_data.fine_tuned_model,
                    )
                    jobs.append(job)

                return jobs

            # Register the provider
            self._providers[FineTuningProvider.OPENAI] = {
                "create_job": openai_create_job,
                "get_job": openai_get_job,
                "list_jobs": openai_list_jobs,
            }

            logger.info("OpenAI fine-tuning provider registered")

        except ImportError:
            logger.info("OpenAI package not installed, skipping provider registration")

    def _register_vertex(self) -> None:
        """Register Vertex AI provider if available."""
        try:
            from google.cloud import aiplatform

            async def vertex_create_job(config: FineTuningConfig) -> FineTuningJob:
                # Initialize Vertex AI
                aiplatform.init()

                # Create a custom job
                job_id = f"ft-job-{int(time.time())}"

                # This is simplified - in reality, you would need to set up a custom
                # job with the appropriate container and parameters for your model

                # For now, just return a placeholder job
                job = FineTuningJob(
                    job_id=job_id,
                    provider=FineTuningProvider.VERTEX,
                    status="pending",
                    created_at=time.time(),
                    base_model=config.base_model,
                )

                return job

            async def vertex_get_job(job_id: str) -> FineTuningJob:
                # Initialize Vertex AI
                aiplatform.init()

                # This is simplified - in reality, you would fetch the job from Vertex AI

                # For now, just return a placeholder job
                job = FineTuningJob(
                    job_id=job_id,
                    provider=FineTuningProvider.VERTEX,
                    status="running",  # Placeholder
                    created_at=time.time() - 3600,  # Placeholder
                    base_model="placeholder",  # Placeholder
                )

                return job

            async def vertex_list_jobs() -> List[FineTuningJob]:
                # Initialize Vertex AI
                aiplatform.init()

                # This is simplified - in reality, you would list jobs from Vertex AI

                # For now, just return an empty list
                return []

            # Register the provider
            self._providers[FineTuningProvider.VERTEX] = {
                "create_job": vertex_create_job,
                "get_job": vertex_get_job,
                "list_jobs": vertex_list_jobs,
            }

            logger.info("Vertex AI fine-tuning provider registered")

        except ImportError:
            logger.info(
                "Google Cloud AI Platform package not installed, skipping provider registration"
            )

    def _register_huggingface(self) -> None:
        """Register Hugging Face provider if available."""
        try:
            # We just check if the package is available
            import importlib.util

            if importlib.util.find_spec("huggingface_hub") is None:
                return

            async def huggingface_create_job(config: FineTuningConfig) -> FineTuningJob:
                # This is simplified - in reality, you would use the Hugging Face
                # API to set up and run a fine-tuning job

                # For now, just return a placeholder job
                job_id = f"hf-ft-{int(time.time())}"

                job = FineTuningJob(
                    job_id=job_id,
                    provider=FineTuningProvider.HUGGINGFACE,
                    status="pending",
                    created_at=time.time(),
                    base_model=config.base_model,
                )

                return job

            async def huggingface_get_job(job_id: str) -> FineTuningJob:
                # This is simplified - in reality, you would fetch the job from Hugging Face

                # For now, just return a placeholder job
                job = FineTuningJob(
                    job_id=job_id,
                    provider=FineTuningProvider.HUGGINGFACE,
                    status="running",  # Placeholder
                    created_at=time.time() - 3600,  # Placeholder
                    base_model="placeholder",  # Placeholder
                )

                return job

            async def huggingface_list_jobs() -> List[FineTuningJob]:
                # This is simplified - in reality, you would list jobs from Hugging Face

                # For now, just return an empty list
                return []

            # Register the provider
            self._providers[FineTuningProvider.HUGGINGFACE] = {
                "create_job": huggingface_create_job,
                "get_job": huggingface_get_job,
                "list_jobs": huggingface_list_jobs,
            }

            logger.info("Hugging Face fine-tuning provider registered")

        except ImportError:
            logger.info(
                "Hugging Face Hub package not installed, skipping provider registration"
            )

    async def create_job(self, config: FineTuningConfig) -> FineTuningJob:
        """
        Create a new fine-tuning job.

        Args:
            config: Configuration for the fine-tuning job

        Returns:
            A FineTuningJob instance representing the created job

        Raises:
            ConfigurationError: If the provider is not supported or properly configured
        """
        if config.provider not in self._providers:
            raise ConfigurationError(
                f"Provider {config.provider} not supported or not properly configured"
            )

        provider_impl = self._providers[config.provider]
        return await provider_impl["create_job"](config)

    async def get_job(
        self, job_id: str, provider: Union[str, FineTuningProvider]
    ) -> FineTuningJob:
        """
        Get the status of a fine-tuning job.

        Args:
            job_id: ID of the job to get
            provider: Provider where the job is running

        Returns:
            A FineTuningJob instance with the current status

        Raises:
            ConfigurationError: If the provider is not supported or properly configured
        """
        provider_enum = (
            FineTuningProvider(provider) if isinstance(provider, str) else provider
        )

        if provider_enum not in self._providers:
            raise ConfigurationError(
                f"Provider {provider} not supported or not properly configured"
            )

        provider_impl = self._providers[provider_enum]
        return await provider_impl["get_job"](job_id)

    async def list_jobs(
        self, provider: Union[str, FineTuningProvider]
    ) -> List[FineTuningJob]:
        """
        List fine-tuning jobs for a provider.

        Args:
            provider: Provider to list jobs for

        Returns:
            List of FineTuningJob instances

        Raises:
            ConfigurationError: If the provider is not supported or properly configured
        """
        provider_enum = (
            FineTuningProvider(provider) if isinstance(provider, str) else provider
        )

        if provider_enum not in self._providers:
            raise ConfigurationError(
                f"Provider {provider} not supported or not properly configured"
            )

        provider_impl = self._providers[provider_enum]
        return await provider_impl["list_jobs"]()

    def supported_providers(self) -> List[FineTuningProvider]:
        """
        Get a list of supported providers.

        Returns:
            List of supported FineTuningProvider values
        """
        return list(self._providers.keys())


class DataConverter:
    """
    Utility for converting between different fine-tuning data formats.

    This class provides methods to convert data between formats supported
    by different fine-tuning providers.
    """

    @staticmethod
    def convert(
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        input_format: Union[str, DataFormat],
        output_format: Union[str, DataFormat],
    ) -> None:
        """
        Convert data from one format to another.

        Args:
            input_file: Path to the input file
            output_file: Path to write the output file
            input_format: Format of the input file
            output_format: Format to convert to

        Raises:
            ConfigurationError: If the conversion is not supported or files are invalid
        """
        input_path = Path(input_file)
        output_path = Path(output_file)

        # Convert string formats to enum
        input_fmt = (
            DataFormat(input_format) if isinstance(input_format, str) else input_format
        )
        output_fmt = (
            DataFormat(output_format)
            if isinstance(output_format, str)
            else output_format
        )

        # Check input file exists
        if not input_path.exists():
            raise ConfigurationError(f"Input file not found: {input_path}")

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if formats are the same
        if input_fmt == output_fmt:
            import shutil

            shutil.copy(input_path, output_path)
            logger.info("Copied file as formats are the same: %s", input_fmt)
            return

        # Load the input data
        data = None
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                if input_fmt == DataFormat.JSONL:
                    data = [json.loads(line) for line in f if line.strip()]
                elif input_fmt in [
                    DataFormat.OPENAI,
                    DataFormat.VERTEX,
                    DataFormat.ALPACA,
                ]:
                    if input_path.suffix.lower() == ".jsonl":
                        data = [json.loads(line) for line in f if line.strip()]
                    else:
                        data = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported input format: {input_fmt}")
        except Exception as e:
            raise ConfigurationError(f"Error reading input file: {str(e)}") from e

        # Convert the data
        converted_data = None

        # Implement conversion logic based on input and output formats
        if input_fmt == DataFormat.OPENAI and output_fmt == DataFormat.VERTEX:
            converted_data = DataConverter._openai_to_vertex(data)
        elif input_fmt == DataFormat.VERTEX and output_fmt == DataFormat.OPENAI:
            converted_data = DataConverter._vertex_to_openai(data)
        elif input_fmt == DataFormat.ALPACA and output_fmt == DataFormat.OPENAI:
            converted_data = DataConverter._alpaca_to_openai(data)
        elif input_fmt == DataFormat.ALPACA and output_fmt == DataFormat.VERTEX:
            converted_data = DataConverter._alpaca_to_vertex(data)
        else:
            raise ConfigurationError(
                f"Conversion from {input_fmt} to {output_fmt} not implemented"
            )

        # Write the output data
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if output_fmt == DataFormat.JSONL or output_fmt == DataFormat.OPENAI:
                    for item in converted_data:
                        f.write(json.dumps(item) + "\n")
                else:
                    json.dump(converted_data, f, indent=2)

            logger.info("Converted data from %s to %s", input_fmt, output_fmt)

        except Exception as e:
            raise ConfigurationError(f"Error writing output file: {str(e)}") from e

    @staticmethod
    def _openai_to_vertex(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI format to Vertex AI format."""
        # This is a simplified conversion
        result = []

        for item in data:
            # OpenAI format: {"messages": [{"role": "...", "content": "..."}, ...]}
            # Vertex format: {"input_text": "...", "output_text": "..."}
            messages = item.get("messages", [])

            # Extract user and assistant messages
            user_message = ""
            assistant_message = ""

            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    assistant_message = msg.get("content", "")

            if user_message and assistant_message:
                result.append(
                    {"input_text": user_message, "output_text": assistant_message}
                )

        return result

    @staticmethod
    def _vertex_to_openai(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Vertex AI format to OpenAI format."""
        # This is a simplified conversion
        result = []

        for item in data:
            # Vertex format: {"input_text": "...", "output_text": "..."}
            # OpenAI format: {"messages": [{"role": "...", "content": "..."}, ...]}
            input_text = item.get("input_text", "")
            output_text = item.get("output_text", "")

            if input_text and output_text:
                result.append(
                    {
                        "messages": [
                            {"role": "user", "content": input_text},
                            {"role": "assistant", "content": output_text},
                        ]
                    }
                )

        return result

    @staticmethod
    def _alpaca_to_openai(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Alpaca format to OpenAI format."""
        # This is a simplified conversion
        result = []

        for item in data:
            # Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
            # OpenAI format: {"messages": [{"role": "...", "content": "..."}, ...]}
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")

            # Combine instruction and input
            user_content = instruction
            if input_text:
                user_content += f"\n\nInput: {input_text}"

            if user_content and output_text:
                result.append(
                    {
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": output_text},
                        ]
                    }
                )

        return result

    @staticmethod
    def _alpaca_to_vertex(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Alpaca format to Vertex AI format."""
        # This is a simplified conversion
        result = []

        for item in data:
            # Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
            # Vertex format: {"input_text": "...", "output_text": "..."}
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")

            # Combine instruction and input
            user_content = instruction
            if input_text:
                user_content += f"\n\nInput: {input_text}"

            if user_content and output_text:
                result.append({"input_text": user_content, "output_text": output_text})

        return result


# Singleton instance
default_fine_tuner = FineTuner()
