"""
Tests verify that payload codec errors and payload converter errors are handled
correctly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Sequence

import nexusrpc
import pytest
from nexusrpc.handler import StartOperationContext, sync_operation

from temporalio import workflow
from temporalio.api.common.v1 import Payload
from temporalio.client import Client, WorkflowFailureError
from temporalio.converter import (
    DataConverter,
    PayloadCodec,
    PayloadConverter,
)
from temporalio.exceptions import ApplicationError, NexusOperationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from tests.helpers.nexus import create_nexus_endpoint, make_nexus_endpoint_name


@dataclass
class Input:
    value: str


@dataclass
class Output:
    value: str


# ============================================================================
# Payload Codecs for testing
# ============================================================================


class PayloadCodecThatRaisesException(PayloadCodec):
    """Payload codec that raises a regular Exception during decode for Nexus operation inputs."""

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return list(payloads)

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        # Check if this looks like the Nexus operation Input dataclass
        # by checking for the JSON content
        for p in payloads:
            if b'"value"' in p.data and b'"test"' in p.data:
                # This is likely the Nexus operation input
                raise Exception("Payload codec decode error")
        return list(payloads)


class PayloadCodecThatRaisesApplicationError(PayloadCodec):
    """Payload codec that raises an ApplicationError during decode for Nexus operation inputs."""

    async def encode(self, payloads: Sequence[Payload]) -> list[Payload]:
        return list(payloads)

    async def decode(self, payloads: Sequence[Payload]) -> list[Payload]:
        # Check if this looks like the Nexus operation Input dataclass
        for p in payloads:
            if b'"value"' in p.data and b'"test"' in p.data:
                # This is likely the Nexus operation input
                raise ApplicationError("Payload codec ApplicationError", non_retryable=True)
        return list(payloads)


# ============================================================================
# Payload Converters for testing
# ============================================================================


class PayloadConverterThatRaisesException(PayloadConverter):
    """Payload converter that raises a regular Exception during from_payloads for specific inputs."""

    def to_payloads(self, values: Sequence[object]) -> list[Payload]:
        # Use default converter for encoding
        return DataConverter.default.payload_converter.to_payloads(values)

    def from_payloads(
        self, payloads: Sequence[Payload], type_hints: list[type] | None = None
    ) -> list[object]:
        # Check if this is the Nexus operation Input
        for p in payloads:
            if b'"value"' in p.data and b'"test"' in p.data:
                raise Exception("Payload converter from_payloads error")
        # Otherwise use default converter
        return DataConverter.default.payload_converter.from_payloads(payloads, type_hints)


class PayloadConverterThatRaisesApplicationError(PayloadConverter):
    """Payload converter that raises an ApplicationError during from_payloads for specific inputs."""

    def to_payloads(self, values: Sequence[object]) -> list[Payload]:
        # Use default converter for encoding
        return DataConverter.default.payload_converter.to_payloads(values)

    def from_payloads(
        self, payloads: Sequence[Payload], type_hints: list[type] | None = None
    ) -> list[object]:
        # Check if this is the Nexus operation Input
        for p in payloads:
            if b'"value"' in p.data and b'"test"' in p.data:
                raise ApplicationError(
                    "Payload converter ApplicationError", non_retryable=True
                )
        # Otherwise use default converter
        return DataConverter.default.payload_converter.from_payloads(payloads, type_hints)


# ============================================================================
# Nexus Service Handlers
# ============================================================================


@nexusrpc.service
class TestService:
    echo_operation: nexusrpc.Operation[Input, Output]


@nexusrpc.handler.service_handler(service=TestService)
class TestServiceHandler:
    @sync_operation
    async def echo_operation(
        self, _ctx: StartOperationContext, input: Input
    ) -> Output:
        return Output(value=f"Processed: {input.value}")


# ============================================================================
# Caller Workflows
# ============================================================================


@workflow.defn
class CallerWorkflow:
    """Workflow that calls a Nexus operation."""

    @workflow.run
    async def run(self, task_queue: str) -> Output:
        nexus_client = workflow.create_nexus_client(
            service=TestServiceHandler,
            endpoint=make_nexus_endpoint_name(task_queue),
        )

        return await nexus_client.execute_operation(
            TestServiceHandler.echo_operation,
            Input(value="test"),
        )


# ============================================================================
# Tests
# ============================================================================


async def test_payload_codec_exception_becomes_internal_handler_error(
    client: Client, env: WorkflowEnvironment
):
    """Test that exceptions from payload codec become INTERNAL HandlerErrors."""
    if env.supports_time_skipping:
        pytest.skip("Nexus tests don't work with time-skipping server")

    task_queue = str(uuid.uuid4())

    # Create client with failing payload codec
    config = client.config()
    config["data_converter"] = DataConverter(
        payload_codec=PayloadCodecThatRaisesException(),
    )
    client = Client(**config)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CallerWorkflow],
        nexus_service_handlers=[TestServiceHandler()],
    ):
        await create_nexus_endpoint(task_queue, client)

        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                CallerWorkflow.run,
                task_queue,
                id=str(uuid.uuid4()),
                task_queue=task_queue,
            )

        # Verify the exception chain
        workflow_error = exc_info.value
        assert isinstance(workflow_error.__cause__, NexusOperationError)

        nexus_error = workflow_error.__cause__
        assert isinstance(nexus_error.__cause__, nexusrpc.HandlerError)

        handler_error = nexus_error.__cause__
        assert handler_error.type == nexusrpc.HandlerErrorType.INTERNAL
        assert "Data converter payload codec failed to decode Nexus operation input" in str(handler_error)
        assert handler_error.retryable_override is False


async def test_payload_codec_application_error_is_reraised(
    client: Client, env: WorkflowEnvironment
):
    """Test that ApplicationErrors from payload codec are re-raised and handled by handler code."""
    if env.supports_time_skipping:
        pytest.skip("Nexus tests don't work with time-skipping server")

    task_queue = str(uuid.uuid4())

    # Create client with failing payload codec
    config = client.config()
    config["data_converter"] = DataConverter(
        payload_codec=PayloadCodecThatRaisesApplicationError(),
    )
    client = Client(**config)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CallerWorkflow],
        nexus_service_handlers=[TestServiceHandler()],
    ):
        await create_nexus_endpoint(task_queue, client)

        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                CallerWorkflow.run,
                task_queue,
                id=str(uuid.uuid4()),
                task_queue=task_queue,
            )

        # Verify the exception chain - ApplicationError should be converted by handler code
        workflow_error = exc_info.value
        assert isinstance(workflow_error.__cause__, NexusOperationError)

        nexus_error = workflow_error.__cause__
        assert isinstance(nexus_error.__cause__, nexusrpc.HandlerError)

        handler_error = nexus_error.__cause__
        # ApplicationError gets converted to INTERNAL handler error
        assert handler_error.type == nexusrpc.HandlerErrorType.INTERNAL
        assert "Payload codec ApplicationError" in str(handler_error.__cause__)


async def test_payload_converter_exception_becomes_bad_request_handler_error(
    client: Client, env: WorkflowEnvironment
):
    """Test that exceptions from payload converter become BAD_REQUEST HandlerErrors."""
    if env.supports_time_skipping:
        pytest.skip("Nexus tests don't work with time-skipping server")

    task_queue = str(uuid.uuid4())

    # Create client with failing payload converter
    config = client.config()
    config["data_converter"] = DataConverter(
        payload_converter_class=PayloadConverterThatRaisesException,
    )
    client = Client(**config)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CallerWorkflow],
        nexus_service_handlers=[TestServiceHandler()],
    ):
        await create_nexus_endpoint(task_queue, client)

        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                CallerWorkflow.run,
                task_queue,
                id=str(uuid.uuid4()),
                task_queue=task_queue,
            )

        # Verify the exception chain
        workflow_error = exc_info.value
        assert isinstance(workflow_error.__cause__, NexusOperationError)

        nexus_error = workflow_error.__cause__
        assert isinstance(nexus_error.__cause__, nexusrpc.HandlerError)

        handler_error = nexus_error.__cause__
        assert handler_error.type == nexusrpc.HandlerErrorType.BAD_REQUEST
        assert "Data converter payload converter failed to decode Nexus operation input" in str(handler_error)
        assert handler_error.retryable_override is False


async def test_payload_converter_application_error_is_reraised(
    client: Client, env: WorkflowEnvironment
):
    """Test that ApplicationErrors from payload converter are re-raised and handled by handler code."""
    if env.supports_time_skipping:
        pytest.skip("Nexus tests don't work with time-skipping server")

    task_queue = str(uuid.uuid4())

    # Create client with failing payload converter
    config = client.config()
    config["data_converter"] = DataConverter(
        payload_converter_class=PayloadConverterThatRaisesApplicationError,
    )
    client = Client(**config)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CallerWorkflow],
        nexus_service_handlers=[TestServiceHandler()],
    ):
        await create_nexus_endpoint(task_queue, client)

        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                CallerWorkflow.run,
                task_queue,
                id=str(uuid.uuid4()),
                task_queue=task_queue,
            )

        # Verify the exception chain - ApplicationError should be converted by handler code
        workflow_error = exc_info.value
        assert isinstance(workflow_error.__cause__, NexusOperationError)

        nexus_error = workflow_error.__cause__
        assert isinstance(nexus_error.__cause__, nexusrpc.HandlerError)

        handler_error = nexus_error.__cause__
        # ApplicationError gets converted to INTERNAL handler error
        assert handler_error.type == nexusrpc.HandlerErrorType.INTERNAL
        assert "Payload converter ApplicationError" in str(handler_error.__cause__)


async def test_both_codec_and_converter_errors(
    client: Client, env: WorkflowEnvironment
):
    """Test that payload codec errors are handled first, before payload converter errors."""
    if env.supports_time_skipping:
        pytest.skip("Nexus tests don't work with time-skipping server")

    task_queue = str(uuid.uuid4())

    # Create client with both failing codec and converter
    config = client.config()
    config["data_converter"] = DataConverter(
        payload_codec=PayloadCodecThatRaisesException(),
        payload_converter_class=PayloadConverterThatRaisesException,
    )
    client = Client(**config)

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[CallerWorkflow],
        nexus_service_handlers=[TestServiceHandler()],
    ):
        await create_nexus_endpoint(task_queue, client)

        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                CallerWorkflow.run,
                task_queue,
                id=str(uuid.uuid4()),
                task_queue=task_queue,
            )

        # Should get codec error (INTERNAL), not converter error (BAD_REQUEST)
        workflow_error = exc_info.value
        assert isinstance(workflow_error.__cause__, NexusOperationError)

        nexus_error = workflow_error.__cause__
        assert isinstance(nexus_error.__cause__, nexusrpc.HandlerError)

        handler_error = nexus_error.__cause__
        assert handler_error.type == nexusrpc.HandlerErrorType.INTERNAL
        assert "payload codec failed to decode" in str(handler_error).lower()
