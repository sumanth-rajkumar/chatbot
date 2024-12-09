from django.test import TestCase
from file_manager.utils import generate_detailed_answer

class GPTTest(TestCase):
    def test_generate_answer(self):
        context = "Synechron’s Client Prospecting Accelerator leverages Big Data and Analytics to automate identifying high-value leads for wealth managers."
        question = "How does Synechron's Client Prospecting Accelerator help wealth managers?"
        answer = generate_detailed_answer(context, question)
        self.assertIn("Big Data and Analytics", answer)
