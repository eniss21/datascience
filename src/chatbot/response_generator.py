import json
import random
import re
from pathlib import Path
from typing import Optional, Dict, List


class ResponseGenerator:
    """Generates responses based on classified intent."""

    def __init__(self, intents_path: str):
        self.intents_path = intents_path
        self.responses: Dict[str, List[str]] = {}
        self._load_responses()

    def _load_responses(self):
        """Load responses from intents file."""
        with open(self.intents_path, 'r') as f:
            data = json.load(f)

        for intent in data['intents']:
            tag = intent['tag']
            self.responses[tag] = intent['responses']

    def get_response(self, intent: str) -> str:
        """Get a random response for the given intent."""
        if intent not in self.responses:
            return self._get_fallback_response()

        responses = self.responses[intent]
        if not responses:
            return self._get_fallback_response()

        return random.choice(responses)

    def _get_fallback_response(self) -> str:
        """Get a fallback response for unknown intents."""
        if 'fallback' in self.responses and self.responses['fallback']:
            return random.choice(self.responses['fallback'])
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

    def extract_order_id(self, text: str) -> Optional[str]:
        """Extract order ID from user message."""
        patterns = [
            r'#?(\d{5,10})',
            r'order\s*#?\s*(\d+)',
            r'ORD-?(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def extract_email(self, text: str) -> Optional[str]:
        """Extract email from user message."""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(pattern, text)
        if match:
            return match.group(0)
        return None

    def format_response(self, intent: str, user_message: str) -> str:
        """Generate formatted response with entity extraction."""
        base_response = self.get_response(intent)

        order_id = self.extract_order_id(user_message)
        if order_id and '{order_id}' in base_response:
            base_response = base_response.replace('{order_id}', order_id)

        return base_response


class Chatbot:
    """Main chatbot class combining intent handling and response generation."""

    def __init__(
        self,
        intents_path: str,
        model_path: Optional[str] = None,
        model_type: str = "tfidf",
        confidence_threshold: float = 0.3
    ):
        from src.chatbot.intent_handler import IntentHandler

        self.intent_handler = IntentHandler(
            model_type=model_type,
            confidence_threshold=confidence_threshold
        )
        self.response_generator = ResponseGenerator(intents_path)
        self.intents_path = intents_path

        if model_path and Path(model_path).exists():
            self.intent_handler.load_model(model_path)
        else:
            self.intent_handler.train_model(intents_path, model_path)

    def chat(self, user_message: str) -> Dict:
        """Process user message and return response."""
        intent, confidence, is_fallback = self.intent_handler.classify_with_fallback(user_message)
        response = self.response_generator.format_response(intent, user_message)

        return {
            "intent": intent,
            "confidence": confidence,
            "is_fallback": is_fallback,
            "response": response
        }

    def get_response(self, user_message: str) -> str:
        """Simple interface - just return the response text."""
        result = self.chat(user_message)
        return result["response"]


if __name__ == "__main__":
    from config import INTENTS_PATH, TFIDF_MODEL_PATH

    print("Initializing chatbot...")
    bot = Chatbot(
        intents_path=str(INTENTS_PATH),
        model_path=str(TFIDF_MODEL_PATH),
        model_type="tfidf"
    )

    print("\nChatbot ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye! Have a great day!")
            break

        result = bot.chat(user_input)
        print(f"Bot: {result['response']}")
        print(f"     [Intent: {result['intent']}, Confidence: {result['confidence']:.2f}]\n")
