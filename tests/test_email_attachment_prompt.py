import ast
from pathlib import Path
import unittest


SOURCE_PATH = Path(__file__).parents[1] / "gtm_agent" / "gtm_agent.py"


def find_definition(tree, name):
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def collect_string_literals(node):
    return " ".join(
        child.value for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    )


class EmailAttachmentPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE_PATH.read_text())
        cls.send_tool = find_definition(cls.tree, "send_prospect_email")
        cls.system_prompt = next(
            node.value for node in cls.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT" for target in node.targets)
        )
        cls.prompt_text = collect_string_literals(cls.system_prompt)

    def test_email_tool_discloses_attachment_limit(self):
        tool_description = ast.get_docstring(self.send_tool)
        self.assertIn("attachments", tool_description)
        self.assertIn("not supported", tool_description)
        self.assertIn("never claim a file is attached", tool_description)
        parameters = {argument.arg for argument in self.send_tool.args.args}
        self.assertFalse(any("attachment" in parameter for parameter in parameters))

    def test_prompt_prohibits_attachment_claims_in_email_body(self):
        self.assertIn("do not claim or imply", self.prompt_text)
        self.assertIn("in the email body", self.prompt_text)
        self.assertIn("that it accompanies the email", self.prompt_text)

    def test_prompt_requires_confirmed_send_and_rep_handoff(self):
        self.assertIn("only report it included in your reply if the send result confirms it", self.prompt_text)
        self.assertIn("the email tool cannot attach or retrieve it", self.prompt_text)
        self.assertIn("not attached and must be sent separately", self.prompt_text)


if __name__ == "__main__":
    unittest.main()
