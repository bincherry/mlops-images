import unittest
from unittest.mock import patch, MagicMock
from translator_and_summarizer import Translator, Summarizer  # 假设您的代码文件名为 translator_and_summarizer


class TestTranslator(unittest.TestCase):
    @patch("translator_and_summarizer.pipeline")
    def setUp(self, mock_pipeline):
        self.mock_pipeline = MagicMock()
        mock_pipeline.return_value = self.mock_pipeline
        self.translator = Translator()

    def test_default_language(self):
        self.assertEqual(self.translator.language, "french", "Default language should be 'french'.")

    @patch("translator_and_summarizer.pipeline")
    def test_translation(self, mock_pipeline):
        mock_pipeline.return_value = lambda text: [{"translation_text": "Bonjour"}]
        result = self.translator.translate("Hello")
        self.assertEqual(result, "Bonjour", "Translation from 'Hello' should be 'Bonjour'.")

    @patch("translator_and_summarizer.pipeline")
    def test_reconfigure_language(self, mock_pipeline):
        self.translator.reconfigure({"language": "german"})
        self.assertEqual(self.translator.language, "german", "Language should be reconfigured to 'german'.")
        mock_pipeline.assert_called_with("translation_en_to_de", model="t5-small")

    @patch("translator_and_summarizer.pipeline")
    def test_invalid_language_reconfigure(self, mock_pipeline):
        self.translator.reconfigure({"language": "invalid_language"})
        self.assertEqual(self.translator.language, "invalid_language", "Language should accept any string.")
        mock_pipeline.assert_not_called()

    def test_empty_reconfigure(self):
        self.translator.reconfigure({})
        self.assertEqual(self.translator.language, "french", "Language should default to 'french' if not specified.")


class TestSummarizer(unittest.TestCase):
    @patch("translator_and_summarizer.pipeline")
    def setUp(self, mock_pipeline):
        self.mock_pipeline = MagicMock()
        mock_pipeline.return_value = self.mock_pipeline
        self.translator_handle = MagicMock()
        self.summarizer = Summarizer(self.translator_handle)

    def test_default_summary_lengths(self):
        self.assertEqual(self.summarizer.min_length, 5, "Default min_length should be 5.")
        self.assertEqual(self.summarizer.max_length, 15, "Default max_length should be 15.")

    @patch("translator_and_summarizer.pipeline")
    def test_summarization(self, mock_pipeline):
        mock_pipeline.return_value = lambda text, min_length, max_length: [
            {"summary_text": "This is a summary."}
        ]
        result = self.summarizer.summarize("This is a long text.")
        self.assertEqual(result, "This is a summary.", "Summarization output mismatch.")

    def test_reconfigure(self):
        self.summarizer.reconfigure({"min_length": 10, "max_length": 20})
        self.assertEqual(self.summarizer.min_length, 10, "Min length should be reconfigured to 10.")
        self.assertEqual(self.summarizer.max_length, 20, "Max length should be reconfigured to 20.")

    @patch("translator_and_summarizer.pipeline")
    @patch("translator_and_summarizer.Translator.translate")
    def test_http_call(self, mock_translate, mock_pipeline):
        mock_pipeline.return_value = lambda text, min_length, max_length: [
            {"summary_text": "This is a summary."}
        ]
        mock_translate.return_value = "Ceci est un résumé."

        mock_request = MagicMock()
        mock_request.json.return_value = "This is a long text."
        result = self.summarizer.__call__(mock_request)

        self.assertIsNotNone(result, "Result should not be None.")
        self.assertEqual(
            result, "Ceci est un résumé.", "HTTP call output mismatch."
        )


if __name__ == "__main__":
    unittest.main()
