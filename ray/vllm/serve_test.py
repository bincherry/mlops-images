import os
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from vllm_deployment import VLLMDeployment, build_app, parse_vllm_args  # 假设您的文件名为 vllm_deployment
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse, ChatCompletionResponse

class TestVLLMDeployment(unittest.TestCase):
    @patch("vllm_deployment.AsyncLLMEngine.from_engine_args")
    def setUp(self, mock_engine):
        mock_engine.return_value = MagicMock()
        engine_args = MagicMock()
        self.deployment = VLLMDeployment(engine_args, response_role="assistant")
        self.client = TestClient(self.deployment.app)

    @patch("vllm_deployment.AsyncLLMEngine.get_model_config")
    @patch("vllm_deployment.OpenAIServingChat.create_chat_completion")
    def test_create_chat_completion(self, mock_create_chat, mock_get_model_config):
        # Mock the engine model configuration
        mock_get_model_config.return_value = {"mock": "config"}

        # Test for ErrorResponse
        mock_create_chat.return_value = ErrorResponse(message="Error", code=400)
        response = self.client.post("/v1/chat/completions", json={"mock": "request"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Error", response.json()["message"])

        # Test for streaming response
        mock_create_chat.return_value = iter(["streamed response part"])
        response = self.client.post("/v1/chat/completions", json={"stream": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream")

        # Test for non-streaming ChatCompletionResponse
        mock_create_chat.return_value = ChatCompletionResponse(message="Success")
        response = self.client.post("/v1/chat/completions", json={"mock": "request"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Success")

    @patch("vllm_deployment.parse_vllm_args")
    def test_build_app(self, mock_parse_args):
        mock_parse_args.return_value = MagicMock()
        os.environ["MODEL_ID"] = "test-model"
        os.environ["SERVED_MODEL_NAME"] = "test-served-model"
        os.environ["TENSOR_PARALLELISM"] = "2"
        app = build_app({})
        self.assertIsNotNone(app, "App should be built successfully.")

    @patch("vllm_deployment.FlexibleArgumentParser")
    def test_parse_vllm_args(self, mock_parser):
        mock_args = MagicMock()
        mock_parser.return_value.parse_args.return_value = mock_args
        cli_args = {"model": "test-model", "served-model-name": "test-name"}
        parsed_args = parse_vllm_args(cli_args)
        self.assertEqual(parsed_args, mock_args, "Parsed args should match mock args.")

    @patch("vllm_deployment.logger.info")
    def test_logging(self, mock_logger):
        engine_args = MagicMock()
        VLLMDeployment(engine_args, response_role="assistant")
        mock_logger.assert_called_with("Starting with engine args: {}".format(engine_args))


if __name__ == "__main__":
    unittest.main()
