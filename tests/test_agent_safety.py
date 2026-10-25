import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

from src.agent import (ConversationState, SupportAgent, _is_damage_question,
                       _is_late_delivery_compensation_request,
                       _has_payment_method, _is_refund_eta_question,
                       _asks_for_human, _is_third_party_contact_request, _is_unauthorized_activity,
                       _needs_order_id, VERIFIED_ORDER_FIELDS)
from src.evaluation_artifacts import SCHEMA, _unsupported_numeric_claims, _write_jsonl
from src.knowledge_base import Chunk


class AgentSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = SupportAgent(use_llm=False)

    def test_answer_prompt_requires_retrieved_evidence(self):
        prompt = self.agent._system_prompt("en")
        self.assertIn("use only the supplied retrieved context", prompt)
        self.assertIn("If evidence is missing", prompt)
        self.assertIn("call create_handover_ticket", prompt)
        self.assertIn("do not escalate a normal policy question", prompt)
        self.assertIn("ask for the missing detail instead of guessing", prompt)
        self.assertIn("opened/sealed status", prompt)
        self.assertIn("first response in each conversation", prompt)

    def test_damage_intent_recognizes_cracks_and_nonworking_products(self):
        self.assertTrue(_is_damage_question("The laptop arrived cracked"))
        self.assertTrue(_is_damage_question("Speaker is not working"))
        self.assertTrue(_is_damage_question("किताब टूटी हुई आई"))
        self.assertFalse(_is_damage_question("Can I return these headphones?"))

    def test_unauthorized_order_intent_is_detected(self):
        self.assertTrue(_is_unauthorized_activity("There is an order I never placed"))
        self.assertTrue(_is_unauthorized_activity("Someone accessed my account"))
        self.assertFalse(_is_unauthorized_activity("Where is my order?"))

    def test_prior_delivery_contact_request_is_sensitive(self):
        self.assertTrue(_is_third_party_contact_request(
            "Give me contact information from a previous delivery attempt"))
        self.assertFalse(_is_third_party_contact_request("What is my own phone number?"))

    def test_specific_order_request_without_id_requires_clarification(self):
        self.assertTrue(_needs_order_id("Mujhe apne order ke baare mein madad chahiye"))
        self.assertTrue(_needs_order_id("Where is my order?"))
        self.assertFalse(_needs_order_id("What is the general refund policy?"))

    def test_missing_order_id_and_third_party_requests_set_decisions(self):
        agent = SupportAgent(use_llm=True)
        with patch.object(agent, "_llm_turn", return_value="Please share your order ID."):
            result = agent.respond("Where is my order?")
        self.assertEqual(result["decision"], "clarify")

        agent = SupportAgent(use_llm=True)
        with patch.object(agent, "_llm_turn", return_value="I cannot share another recipient's contact details."):
            result = agent.respond("Give me contact information from a previous delivery attempt")
        self.assertEqual(result["decision"], "refuse")

    def test_refund_eta_clarifies_missing_payment_method(self):
        self.assertTrue(_is_refund_eta_question("When will my refund come?"))
        self.assertFalse(_has_payment_method("When will my refund come?"))
        self.assertTrue(_has_payment_method("UPI refund, how long?"))

        agent = SupportAgent(use_llm=True)
        with patch.object(agent, "_llm_turn", return_value="Which payment method did you use?"):
            result = agent.respond("When will my refund come?")
        self.assertEqual(result["decision"], "clarify")

    def test_explicit_human_request_is_detected(self):
        self.assertTrue(_asks_for_human("Can I talk to a person?"))
        self.assertTrue(_asks_for_human("Please connect me to a live agent"))
        self.assertFalse(_asks_for_human("How do I reach the refund team?"))

    def test_late_delivery_compensation_requires_policy_refusal_for_standard_order(self):
        self.assertTrue(_is_late_delivery_compensation_request("Order late aaya, mujhe 300 rupaye ka voucher chahiye"))
        self.assertFalse(_is_late_delivery_compensation_request("The order arrived late"))

        agent = SupportAgent(use_llm=True)
        agent.state = ConversationState(verified_order={"order_id": "12345678", "delivery_type": "standard"})
        with patch.object(agent, "_llm_turn", return_value="I can't provide a voucher for this order."):
            result = agent.respond("Order late aaya, mujhe 300 rupaye ka voucher chahiye")
        self.assertEqual(result["decision"], "refuse")

    def test_numeric_answer_claims_must_appear_in_retrieved_evidence(self):
        kb = type("FakeKB", (), {"chunks": [Chunk("policy.md", "REF-2", "Refund timing",
                                                     "Refunds take 5 to 7 business days.", {})]})()
        unsupported = _unsupported_numeric_claims(
            "Your refund will arrive in 30 days.", ["policy.md:REF-2"], {}, kb,
            self.agent.data)
        self.assertTrue(any("30" in claim for claim in unsupported))
        supported = _unsupported_numeric_claims(
            "Refunds take 5 to 7 business days.", ["policy.md:REF-2"], {}, kb,
            self.agent.data)
        self.assertEqual(supported, [])

    def test_language_validation_does_not_erase_retrieval_trace(self):
        agent = SupportAgent(use_llm=True)

        def wrong_language(_message):
            agent.state.evidence = ["pol_returns_v4.md:RET-2"]
            return "The return window is 30 days."

        with patch.object(agent, "_llm_turn", side_effect=wrong_language), \
                patch.object(agent.llm, "chat", return_value="माफ़ कीजिए, जानकारी उपलब्ध है।"):
            result = agent.respond("क्या मैं सामान वापस कर सकता हूँ?")
        self.assertEqual(result["sources"], ["pol_returns_v4.md:RET-2"])
        self.assertEqual(result["retrieved_evidence"], ["pol_returns_v4.md:RET-2"])

    def test_order_details_require_customer_provided_verification(self):
        order_id, order = next(iter(self.agent.data.orders.items()))
        customer = self.agent.data.customers[order["customer_id"]]
        proof = customer["phone"][-4:]
        self.agent.state = ConversationState(order_id=order_id)

        first = self.agent._call_tool("lookup_order", {"order_id": order_id}, "Order " + order_id)
        self.assertTrue(first.get("verification_required"))
        self.assertNotIn("status", first)

        guessed = self.agent._call_tool("lookup_order", {"order_id": order_id,
                                                           "verification_value": "0000"},
                                        "Order " + order_id + ", last four are 0000")
        self.assertFalse(guessed.get("verified"))
        self.assertNotIn("status", guessed)
        self.assertFalse(self.agent.state.verified_order)

        verified = self.agent._call_tool("lookup_order", {"order_id": order_id,
                                                            "verification_value": proof},
                                         "Last four digits " + proof)
        self.assertTrue(verified.get("verified"))
        self.assertTrue(self.agent.state.verified_order)

    def test_tool_uses_full_customer_pincode_when_model_truncates_it(self):
        agent = SupportAgent(use_llm=False)
        agent.state.order_id = "12345678"
        with patch("tools.tools.lookup_order", return_value={"verified": True, "order_id": "12345678"}) as lookup:
            result = agent._call_tool("lookup_order", {
                "order_id": "12345678", "verification_value": "7000"
            }, "Order 12345678, delivery pincode 700091")
        self.assertTrue(result["verified"])
        self.assertEqual(lookup.call_args.kwargs["verification_value"], "700091")

    def test_tool_uses_full_customer_pincode_when_model_truncates_it(self):
        agent = SupportAgent(use_llm=False)
        agent.state.order_id = "12345678"
        with patch("tools.tools.lookup_order", return_value={"verified": True, "order_id": "12345678"}) as lookup:
            result = agent._call_tool("lookup_order", {
                "order_id": "12345678", "verification_value": "7000"
            }, "Order 12345678, delivery pincode 700091")
        self.assertTrue(result["verified"])
        self.assertEqual(lookup.call_args.kwargs["verification_value"], "700091")

    def test_new_order_id_clears_previous_verification_state(self):
        agent = SupportAgent(use_llm=False)
        agent.state = ConversationState(order_id="12345678", verified_order={"order_id": "12345678"},
                                        verification_attempts=1,
                                        previous_tool_results=[{"name": "lookup_order", "result": {}}])
        agent._remember_order_id("Please check order 87654321")
        self.assertEqual(agent.state.order_id, "87654321")
        self.assertIsNone(agent.state.verified_order)
        self.assertEqual(agent.state.verification_attempts, 0)
        self.assertEqual(agent.state.previous_tool_results, [])

    def test_dates_are_not_misread_as_order_ids(self):
        agent = SupportAgent(use_llm=False)
        agent._remember_order_id("Refund started on 2026-10-17")
        self.assertIsNone(agent.state.order_id)
        agent._remember_order_id("Order number 8978 2837")
        self.assertEqual(agent.state.order_id, "89782837")

    def test_date_numbers_are_not_misread_as_verification_proof(self):
        agent = SupportAgent(use_llm=False)
        agent.state.order_id = "12345678"
        self.assertIsNone(agent._verification_value("Refund initiated on 2026-10-17"))
        self.assertIsNone(agent._verification_value("Refund started on 17 October 2026"))
        self.assertEqual(agent._verification_value("Last four digits are 4533"), "4533")
        self.assertEqual(agent._verification_value("Delivery pincode 560001"), "560001")

    def test_verified_order_context_exposes_shipping_date_but_not_verification_pincode(self):
        self.assertIn("shipped_on", VERIFIED_ORDER_FIELDS)
        self.assertIn("contacts_last_7_days", VERIFIED_ORDER_FIELDS)
        self.assertNotIn("delivery_pincode", VERIFIED_ORDER_FIELDS)
        self.assertNotIn("phone", VERIFIED_ORDER_FIELDS)

    def test_ticket_only_marks_created_when_tool_confirms(self):
        self.agent.state = ConversationState()
        with patch("tools.tools.create_ticket", return_value={"created": False, "error": "unavailable"}):
            failed = self.agent._call_tool("create_handover_ticket", {"summary": "Customer needs a human.",
                                                                        "reason": "requested"}, "Please connect me")
        self.assertFalse(failed["created"])
        self.assertFalse(self.agent.state.ticket_created)
        with patch("tools.tools.create_ticket", return_value={"created": True, "ticket_id": "TKT-TEST"}):
            created = self.agent._call_tool("create_handover_ticket", {"summary": "Customer needs a human.",
                                                                          "reason": "requested"}, "Please connect me")
        self.assertTrue(created["created"])
        self.assertTrue(self.agent.state.ticket_created)

    def test_business_day_tool_handles_missing_arguments_and_excludes_sunday(self):
        self.agent.state = ConversationState()
        missing = self.agent._call_tool("calculate_business_day_range", {
            "start_date": "2026-10-05", "minimum_days": 1, "maximum_days": None
        }, "refund business days")
        self.assertIn("error", missing)
        calculated = self.agent._call_tool("calculate_business_day_range", {
            "start_date": "2026-10-05", "minimum_days": 1, "maximum_days": 2
        }, "business days")
        self.assertEqual(calculated["minimum_date"], "2026-10-06")
        self.assertEqual(calculated["maximum_date"], "2026-10-07")

    def test_date_parser_accepts_ordinal_dates_with_of(self):
        parsed = SupportAgent._delivery_date_from_text("purchased during the sale on 5th of October 2026")
        self.assertEqual(parsed, date(2026, 10, 5))

    def test_conversation_state_is_isolated_per_agent(self):
        first = SupportAgent(use_llm=False)
        second = SupportAgent(use_llm=False)
        first.state.order_id = "12345678"
        first.state.verified_order = {"order_id": "12345678"}
        first.state.conversation_history.append({"user": "hello", "assistant": "hi"})
        self.assertIsNone(second.state.order_id)
        self.assertIsNone(second.state.verified_order)
        self.assertEqual(second.state.conversation_history, [])

    def test_artifact_writer_enforces_the_exact_13_field_schema(self):
        record = dict.fromkeys(SCHEMA)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "results.jsonl")
            _write_jsonl(__import__("pathlib").Path(path), [record], enforce_schema=True)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(list(__import__("json").loads(handle.read())), SCHEMA)
            with self.assertRaises(ValueError):
                _write_jsonl(__import__("pathlib").Path(path), [{"unexpected": True}], enforce_schema=True)


if __name__ == "__main__":
    unittest.main()
