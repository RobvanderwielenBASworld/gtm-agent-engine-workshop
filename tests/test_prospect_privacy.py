import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_core.callbacks import BaseCallbackHandler

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect


SENSITIVE_KEYS = (
    "billing_qualification", "tax_id", "date_of_birth", "card_on_file",
    "credit_check_ref",
)
SENSITIVE_VALUES = (
    "tax-secret", "dob-secret", "card-secret", "credit-secret",
)


class ToolInputCapture(BaseCallbackHandler):
    def __init__(self):
        self.inputs = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.inputs.append(kwargs.get("inputs"))


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()
        self.prospect_id = "LEAD-TEST"
        self.record = {
            "prospect_id": self.prospect_id,
            "name": "Taylor Prospect",
            "email": "taylor@example.com",
            "annual_revenue": 42000000,
            "billing_qualification": {
                "tax_id": "tax-secret",
                "date_of_birth": "dob-secret",
                "card_on_file": "card-secret",
                "credit_check_ref": "credit-secret",
            },
            "metadata": [{"tax_id": "tax-secret"}],
        }

    def assert_no_sensitive_data(self, value):
        serialized = json.dumps(value, default=str)
        for sensitive in SENSITIVE_KEYS + SENSITIVE_VALUES:
            self.assertNotIn(sensitive, serialized)

    def test_contact_lookup_returns_only_contact_fields(self):
        with patch.object(data_service, "get_prospect_record", return_value=self.record):
            result = get_prospect.invoke({"prospect_id": self.prospect_id})

        self.assertEqual(result, {
            "prospect": {
                "prospect_id": self.prospect_id,
                "name": "Taylor Prospect",
                "email": "taylor@example.com",
            },
            "found": True,
        })
        self.assert_no_sensitive_data(result)

    def test_profile_build_and_storage_exclude_billing_fields(self):
        with (
            patch.object(data_service, "get_prospect_record", return_value=self.record),
            patch.object(data_service, "fetch_engagement_history", return_value=["call"]),
            patch.object(data_service, "fetch_account_details", return_value={
                "region": "West", "billing_qualification": {"card_on_file": "card-secret"},
            }),
            patch.object(data_service, "fetch_tech_stack", return_value=["Snowflake"]),
        ):
            result = build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        profile = result["prospect_profile"]
        self.assertEqual(profile["name"], "Taylor Prospect")
        self.assertEqual(profile["annual_revenue"], 42000000)
        self.assertEqual(profile["engagement_history"], ["call"])
        self.assertEqual(profile["account_details"], {"region": "West"})
        self.assertEqual(profile["tech_stack"], ["Snowflake"])
        self.assertEqual(data_service._PROFILES[self.prospect_id], profile)
        self.assert_no_sensitive_data(result)
        self.assert_no_sensitive_data(data_service._PROFILES[self.prospect_id])

    def test_cached_profile_is_sanitized_before_return(self):
        data_service._PROFILES[self.prospect_id] = dict(self.record)

        result = build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertEqual(result["prospect_profile"]["name"], "Taylor Prospect")
        self.assertNotIn("billing_qualification", data_service._PROFILES[self.prospect_id])
        self.assert_no_sensitive_data(result)

    def test_profile_storage_sanitizes_nested_sensitive_fields(self):
        data_service.save_profile_to_db(self.prospect_id, self.record)

        self.assert_no_sensitive_data(data_service._PROFILES[self.prospect_id])

    def test_scoring_trace_and_prompt_use_allowlisted_projection(self):
        profile = {
            **self.record,
            "tech_stack": ["Snowflake"],
            "account_details": {
                "region": "West",
                "billing_qualification": {"tax_id": "tax-secret"},
            },
            "engagement_history": ["call"],
            "disqualified": True,
            "enrichment_source": "private-source",
        }
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 1000000,
            "description": "Test offering",
        }
        capture = ToolInputCapture()
        llm_result = Mock(model_dump=lambda: {"score": 80})
        scoring_llm = Mock()
        scoring_llm.invoke.return_value = llm_result

        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Snowflake"]),
            patch("gtm_agent.gtm_agent._scoring_llm", scoring_llm),
        ):
            result = score_prospect.invoke(
                {
                    "prospect_profile": profile,
                    "offering": offering,
                    "tax_id": "tax-secret",
                },
                config={"callbacks": [capture]},
            )

        self.assertEqual(result, {"score": 80})
        self.assertEqual(len(capture.inputs), 1)
        self.assertEqual(set(capture.inputs[0]), {"prospect_profile", "offering"})
        traced_input = capture.inputs[0]["prospect_profile"]
        expected_fields = {
            "prospect_id", "name", "annual_revenue", "tech_stack",
            "account_details", "engagement_history",
        }
        self.assertEqual(set(traced_input), expected_fields)
        self.assertEqual(traced_input["name"], "Taylor Prospect")
        self.assertEqual(traced_input["annual_revenue"], 42000000)
        self.assertEqual(traced_input["tech_stack"], ["Snowflake"])
        self.assert_no_sensitive_data(capture.inputs)
        prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        prompt_profile = json.loads(prompt.split("Prospect profile:\n", 1)[1])
        self.assertEqual(set(prompt_profile), expected_fields)
        self.assertEqual(prompt_profile["annual_revenue"], 42000000)
        self.assertEqual(prompt_profile["tech_stack"], ["Snowflake"])
        self.assertNotIn("disqualified", prompt)
        self.assertNotIn("enrichment_source", prompt)
        self.assert_no_sensitive_data(prompt)


if __name__ == "__main__":
    unittest.main()
