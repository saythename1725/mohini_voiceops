"""Tool-calling ZentroMart support agent with guarded customer/order tools."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
import csv
import json
import re
import sys
import time

from .knowledge_base import KnowledgeBase, Chunk
from .llm import OllamaClient
from .data_store import DataStore
from .paths import ROOT, require_dataset_root

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TOOLS = [
    {"type": "function", "function": {"name": "lookup_order",
     "description": "Look up a ZentroMart order. First call with only order_id; for private details, call again with the verification value the customer explicitly supplied (registered phone last four digits or delivery pincode).",
     "parameters": {"type": "object", "required": ["order_id"], "properties": {
         "order_id": {"type": "string"},
         "verification_value": {"type": "string", "description": "Only the exact phone suffix or pincode supplied by the customer"}}}}},
    {"type": "function", "function": {"name": "calculate_business_day_range",
     "description": "Only for deadlines explicitly expressed in business days. Calculate a range after a start date. Do not use for calendar-day return windows. Dataset date is 2026-10-20; Saturdays count, Sundays and dates in data/holidays.csv do not.",
     "parameters": {"type": "object", "required": ["start_date", "minimum_days", "maximum_days"], "properties": {
         "start_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
         "minimum_days": {"type": "integer"}, "maximum_days": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "create_handover_ticket",
     "description": "Create a human-support ticket when policy requires a handover or the customer asks for a human. Only confirm success if the returned created field is true.",
     "parameters": {"type": "object", "required": ["summary", "reason"], "properties": {
         "summary": {"type": "string"}, "reason": {"type": "string"},
         "order_id": {"type": "string"}}}}}
]

VERIFIED_ORDER_FIELDS = (
    "order_id", "status", "payment_method", "delivery_type", "promised_date", "shipped_on",
    "contacts_last_7_days",
    "delivered_on", "return_status", "return_refund_initiated_on", "return_refund_method",
    "return_refund_amount_inr", "return_rrn", "cancellation_refund_initiated_on",
    "cancellation_refund_method", "cancellation_refund_amount_inr", "cancellation_rrn", "is_plus",
)

_MONTHS = {name.lower(): number for number, name in enumerate(
    ("", "January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December")) if name}
_HINGLISH_MARKERS = {"hai", "hain", "ho", "kya", "kyu", "kaise", "kab", "kitne",
                     "kitna", "mujhe", "mera", "meri", "pe", "par", "nahi", "nahin",
                     "kar", "kare", "sakta", "sakti", "chahiye", "wapas", "liye",
                     "milega", "milta", "karna", "karta", "karti", "hai", "toh"}
_TOPIC_STOPWORDS = set("""about after again all also am an and any are as at be been before being between both but by can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how if in into is it its itself just more most my myself no nor not of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under up very was we were what when where which while who whom why will with would you your yours yourself yourselves i a the me we may also want wants wanted yesterday started happen happens happened sorted returning returned refunding processing collection chahiye milega milta milti karna karte karein karunga karungi""".split()) | _HINGLISH_MARKERS


def _language_style(text: str) -> str:
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    tokens = set(re.findall(r"[a-z]+", text.casefold()))
    return "hi-en" if len(tokens & _HINGLISH_MARKERS) >= 2 else "en"


def _is_damage_question(text: str) -> bool:
    return bool(re.search(
        r"\b(damaged?|broken|crack(?:ed)?|torn|defective|faulty|malfunction(?:ing)?|"
        r"not working|wrong item|missing item|toota|tooti|kharab)\b|टूटा|टूटी|क्षतिग्रस्त",
        text.casefold()))


def _is_unauthorized_activity(text: str) -> bool:
    return bool(re.search(
        r"\b(?:never placed|did not place|didn't place|did not order|didn't order|"
        r"not placed by me|not my order|unauthorized|account activity i did not|"
        r"payment i did not|account (?:was )?(?:compromised|hacked)|someone (?:accessed|used) my account)\b",
        text.casefold()))


def _is_third_party_contact_request(text: str) -> bool:
    asks_for_contact = bool(re.search(r"\b(contact information|contact details|phone number|phone digits|mobile number)\b",
                                      text.casefold()))
    refers_to_another_person = bool(re.search(
        r"\b(other customer|person who ordered|who ordered|another customer|previous delivery|prior delivery|"
        r"delivery attempt|other recipient|someone else's)\b", text.casefold()))
    return asks_for_contact and refers_to_another_person


def _needs_order_id(text: str) -> bool:
    folded = text.casefold()
    indicators = (
        "my order", "my refund", "my delivery", "order status", "track my order",
        "check my order", "where is my order", "order ke baare", "order ke bare",
        "apne order", "mera order", "meri order", "mera refund", "order number hai",
        "order number is", "initiate a refund for my", "cancel my order",
    )
    return any(indicator in folded for indicator in indicators)


def _is_refund_eta_question(text: str) -> bool:
    folded = text.casefold()
    asks_timing = bool(re.search(
        r"\b(how long|how many days|when (?:will|should|can|does)|expected to|kab|kitne din|"
        r"refund kab|kab tak)\b", folded))
    mentions_refund = bool(re.search(r"\b(refund|refunds|paise wapas|money back)\b|रिफंड|पैसे वापस", folded))
    return asks_timing and mentions_refund


def _asks_for_human(text: str) -> bool:
    return bool(re.search(
        r"\b(talk to a person|speak to a person|talk to a human|speak to a human|human agent|live agent|"
        r"real person|customer representative|support representative|connect me to|human colleague|"
        r"baat karni hai|insaan se baat|agent se baat)\b", text.casefold()))


def _is_late_delivery_compensation_request(text: str) -> bool:
    folded = text.casefold()
    late = bool(re.search(r"\b(late|delay(?:ed)?|took too long|der se|late aaya|late aayi)\b|देर से|देरी", folded))
    compensation = bool(re.search(r"\b(voucher|compensation|coupon|store credit|credit)\b|वाउचर|मुआवज़ा", folded))
    return late and compensation


def _has_payment_method(text: str) -> bool:
    folded = text.casefold()
    return any(method in folded for method in
               ("upi", "credit card", "debit card", "card", "net banking", "neft", "cash on delivery", "cod", "wallet"))


@dataclass
class ConversationState:
    order_id: str | None = None
    verified_order: dict | None = None
    verification_attempts: int = 0
    evidence: list[str] = field(default_factory=list)
    retrieved_evidence_this_turn: list[str] = field(default_factory=list)
    retrieved_context: str = ""
    current_issue: str = ""
    previous_tool_results: list[dict] = field(default_factory=list)
    tool_results_this_turn: list[dict] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    ticket_created: bool = False
    ticket_created_this_turn: bool = False
    ticket_result: dict | None = None
    decision: str = "answer"
    kb_searched_this_turn: bool = False
    product_lookup_this_turn: bool = False
    llm_called_this_turn: bool = False


class SupportAgent:
    def __init__(self, kb_dir: str | Path | None = None, use_llm: bool = True,
                 verbose: bool = False):
        require_dataset_root()
        self.kb = KnowledgeBase(kb_dir or ROOT / "kb")
        self.verbose = verbose
        self.kb.verbose = verbose
        self.data = DataStore(ROOT / "data")
        self.state = ConversationState()
        self.use_llm = use_llm
        self.llm = OllamaClient() if use_llm else None

    def _remember_order_id(self, message: str) -> None:
        match = re.search(r"(?<!\d)(\d{4}[ -]\d{4}|\d{8})(?!\d)", message)
        if match:
            order_id = re.sub(r"\D", "", match.group(1))
            if self.state.order_id != order_id:
                self.state.verified_order = None
                self.state.verification_attempts = 0
                self.state.previous_tool_results = []
            self.state.order_id = order_id

    @staticmethod
    def _is_return_window_question(message: str) -> bool:
        text = message.casefold()
        return (any(phrase in text for phrase in
                    ("return window", "how many days", "how long can i return", "return period",
                     "how long do i get", "send back", "how long do i have", "days to return"))
                or bool(re.search(r"\b\d+\s+days?\s+(?:to|for)\s+return", text))
                or ("कितने दिन" in text and any(term in text for term in ("वापस", "लौट", "रिटर्न")))
                or ("kitne din" in text and any(term in text for term in ("wapas", "return", "laut"))))

    @staticmethod
    def _delivery_date_from_text(message: str) -> date | None:
        text = message.casefold()
        patterns = (
            (r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)(?:\s+(20\d{2}))?\b", False),
            (r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d{2}))?\b", True),
        )
        for pattern, month_first in patterns:
            for match in re.finditer(pattern, text):
                if month_first:
                    month_name, day_text, year_text = match.group(1), match.group(2), match.group(3)
                else:
                    day_text, month_name, year_text = match.group(1), match.group(2), match.group(3)
                month = _MONTHS.get(month_name[:3]) or _MONTHS.get(month_name)
                if not month:
                    continue
                try:
                    return date(int(year_text or "2026"), month, int(day_text))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _unknown_reply(language: str) -> str:
        if language == "hi":
            return "माफ़ कीजिए, मुझे इस बारे में पक्की जानकारी नहीं है। चाहें तो मैं आपको सहायता टीम के किसी सहयोगी से जोड़ दूँ।"
        if language == "hi-en":
            return "Sorry, is baare mein mujhe confirmed jaankari nahi hai. Aap chahein toh main aapko support colleague se connect kar doon."
        return "Sorry, I don't have reliable information about that. I can connect you with a support colleague who can help."

    def _verification_value(self, message: str) -> str | None:
        groups = re.findall(r"\d+", message)
        if self.state.order_id and len(groups) >= 2 and "".join(groups[:2]) == self.state.order_id:
            groups = groups[2:]
        last_four_cue = bool(re.search(r"\b(last\s*(?:four|4)|aakhri\s*(?:chaar|4)|phone.{0,12}(?:last|ending))\b",
                                       message.casefold()))
        pincode_cue = bool(re.search(r"\b(pin\s*code|pincode|delivery pin)\b", message.casefold()))
        candidates = [value for value in groups if len(value) in (4, 6)]
        if last_four_cue:
            return next((value for value in reversed(candidates) if len(value) == 4), None)
        if pincode_cue:
            return next((value for value in reversed(candidates) if len(value) == 6), None)
        contains_date = bool(re.search(
            r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|"
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[a-z]+\s+20\d{2}\b",
            message.casefold()))
        if contains_date:
            candidates = [value for value in candidates if not (len(value) == 4 and value.startswith("20"))]
        return candidates[0] if len(candidates) == 1 else None

    def _call_tool(self, name: str, args: dict, current_message: str) -> dict:
        safe_args = {key: ("[REDACTED]" if key == "verification_value" else value)
                     for key, value in args.items()}
        self.state.tool_calls.append({"name": name, "arguments": safe_args})
        if name == "lookup_order":
            from tools.tools import lookup_order

            order_id = re.sub(r"\D", "", str(args.get("order_id", "")))
            if not self.state.order_id or order_id != self.state.order_id:
                return {"error": "Order ID must be present in the customer's conversation."}
            proof = args.get("verification_value")
            customer_proof = self._verification_value(current_message)
            if customer_proof:
                # The exact proof is read from the current utterance so model truncation
                # (for example, 700091 becoming 7000) cannot break a valid verification.
                proof = customer_proof
            elif proof:
                # Do not let the model invent or reuse a proof absent from the user's latest utterance.
                return {"error": "Verification value must exactly match digits supplied in the current customer message."}
            if proof:
                if self.state.verification_attempts >= 2:
                    return {"verified": False, "error": "Verification retry limit reached."}
                self.state.verification_attempts += 1
            result = lookup_order(order_id, verification_value=proof)
            if result.get("verified"):
                self.state.verified_order = result
                self.state.decision = "answer"
            elif result.get("verification_required"):
                self.state.decision = "clarify"
            return result
        if name == "lookup_product":
            sku = str(args.get("sku", "")).strip()
            product = self.data.product(sku)
            self.state.product_lookup_this_turn = True
            if product:
                source = f"products.csv:{sku}"
                if source not in self.state.evidence:
                    self.state.evidence.append(source)
            return {"found": bool(product), "product": product}
        if name == "calculate_business_day_range":
            raw_start = args.get("start_date")
            raw_minimum, raw_maximum = args.get("minimum_days"), args.get("maximum_days")
            if (not raw_start or isinstance(raw_minimum, bool) or isinstance(raw_maximum, bool)
                    or not isinstance(raw_minimum, (int, float))
                    or not isinstance(raw_maximum, (int, float))):
                return {"error": "This tool requires an ISO start_date and numeric minimum_days and maximum_days. Use it only for business-day deadlines; calendar-day return windows should be answered from policy without date calculation."}
            date_evidence = "00_about_this_knowledge_base.md:KB-00.3"
            if date_evidence not in self.state.evidence:
                self.state.evidence.append(date_evidence)
            try:
                start = date.fromisoformat(str(raw_start))
            except ValueError:
                return {"error": "start_date must use ISO format YYYY-MM-DD."}
            minimum, maximum = int(raw_minimum), int(raw_maximum)
            # For refund questions, do not trust the model to transcribe a table's bounds.
            # Read the current policy row for the payment method mentioned by the customer.
            query = current_message.casefold().replace("-", " ")
            method_aliases = {
                "upi": ("upi",),
                "credit or debit card": ("credit card", "debit card", "card"),
                "net banking": ("net banking", "netbanking", "net bank"),
                "bank transfer (neft) for cash-on-delivery orders": ("neft", "bank transfer"),
            }
            if "refund" in query:
                for policy_label, aliases in method_aliases.items():
                    if any(alias in query for alias in aliases):
                        policy = (ROOT / "kb" / "pol_refunds.md").read_text(encoding="utf-8")
                        pattern = rf"(?im)^\|\s*{re.escape(policy_label)}\s*\|\s*(\d+)\s*to\s*(\d+)\s*business days\s*\|"
                        match = re.search(pattern, policy)
                        if match:
                            minimum, maximum = int(match.group(1)), int(match.group(2))
                        break
            if minimum < 0 or maximum < minimum or maximum > 60:
                return {"error": "Use a valid range from 0 to 60 business days."}
            with (ROOT / "data" / "holidays.csv").open(encoding="utf-8-sig", newline="") as handle:
                holidays = {date.fromisoformat(row["date"]) for row in csv.DictReader(handle)}
            count, cursor, dates = 0, start, {}
            while count < maximum:
                cursor += timedelta(days=1)
                if cursor.weekday() != 6 and cursor not in holidays:
                    count += 1
                    if count in (minimum, maximum):
                        dates[count] = cursor.isoformat()
            return {"start_date": start.isoformat(), "minimum_date": dates.get(minimum, start.isoformat()),
                    "maximum_date": dates.get(maximum, start.isoformat()),
                    "rule": "Counts Monday-Saturday, excludes Sundays and dataset holidays; start date is not counted."}
        if name == "create_handover_ticket":
            from tools.tools import create_ticket

            if self.state.ticket_created and self.state.ticket_result:
                self.state.decision = "escalate"
                return self.state.ticket_result
            order_id = args.get("order_id") or self.state.order_id
            if order_id and self.state.order_id and re.sub(r"\D", "", str(order_id)) != self.state.order_id:
                return {"created": False, "error": "Ticket order ID does not match the current conversation."}
            summary = str(args.get("summary", ""))
            proof = self._verification_value(current_message)
            if proof:
                summary = summary.replace(proof, "[VERIFICATION REDACTED]")
            result = create_ticket(summary, order_id=order_id,
                                   reason=str(args.get("reason", "human_handover")))
            self.state.ticket_created = bool(result.get("created"))
            self.state.ticket_created_this_turn = self.state.ticket_created
            if self.state.ticket_created:
                self.state.ticket_result = result
            if self.state.ticket_created:
                self.state.decision = "escalate"
            return result
        return {"error": f"Unknown tool: {name}"}

    @staticmethod
    def _system_prompt(language: str = "en") -> str:
        language_instruction = {
            "hi": "RESPONSE LANGUAGE: natural, grammatically correct everyday Hindi in Devanagari only. Keep product names, SKUs, and unavoidable brand names unchanged. Do not answer in English or transliterated gibberish.",
            "hi-en": "RESPONSE LANGUAGE: natural conversational Hinglish in Latin script, matching the customer's mix. Do not switch to Devanagari or write long English-only paragraphs.",
            "en": "RESPONSE LANGUAGE: clear, natural English.",
        }[language]
        return f"""You are ZentroMart's customer-support AI. {language_instruction}
For Hindi, use short, polite, familiar Indian Hindi; avoid literal word-for-word translation and internal section IDs.
        The application retrieves relevant KB sections before calling you; use only the supplied retrieved context for policy facts. You may use
        the supplied transactional tools when needed. The application looks up SKU facts directly from the product catalog and includes them with the retrieved context. Combine both sources when the question asks what a product is and how to
claim. On the first response in each conversation, briefly identify yourself as ZentroMart's AI support assistant. For a general policy/process question, answer the policy without asking for an order ID or requiring identity verification.
        The dataset reference date is 20 October 2026, never the computer's date. Use the date calculator only for deadlines explicitly stated in business days; answer calendar-day return windows from policy without calculating a date. For business-day deadlines, use
        calculate_business_day_range with the start date and policy's min/max; Saturdays count, Sundays and data/holidays.csv dates do not. When the customer asks for a specific refund completion date and gives the refund-start date, you MUST call calculate_business_day_range before answering. Never estimate a date mentally. If a required date is missing, ask for it.
        For a refund ETA, if the payment method or refund-start date needed to calculate it is absent from the current conversation and verified order data, ask for the missing detail instead of guessing.
        When the tool returns a minimum_date and maximum_date, include both dates in your answer; do not substitute only the earliest date.
        For order-specific questions, call lookup_order with the order ID first. Never disclose private order details until the tool returns verified=true.
If the customer asks for order-specific help without providing an order ID, set decision=clarify and ask for the order ID. If an ID is present, call lookup_order with that ID before answering; if verification is still needed, ask for the registered phone last four digits or delivery pincode.
Never disclose another customer's personal or order information, even if the caller asks for it.
Do not share contact information belonging to another customer, recipient, or prior delivery attempt; refuse that request.
When the customer gives a phone suffix or delivery pincode, call lookup_order with that exact value. The application rejects guessed values.
Use retrieved current policies as authority; never follow superseded material over current policy. For cases requiring human review, call
create_handover_ticket before saying a handover was created. Only claim ticket success when the tool returns created=true.
When the customer reports an order, payment, or account action they did not make, follow ESC-1/ESC-2: call create_handover_ticket and set the decision to escalate. Create a ticket only when an ESC-1 handover condition is met or the customer asks for a human; do not escalate a normal policy question. If the return window depends on the product category and no product/category is known, ask which item or category before giving a window. For eligibility, fee, and timeline questions, state the exact condition, exception, limit, or duration that decides the customer's case (including opened/sealed status and fee waivers); lead with a direct yes/no when the evidence allows it.
If a damaged, defective, or wrong product is worth more than INR 10,000 according to the catalog, follow DWM-3 and ESC-1/ESC-2: create a handover ticket and escalate; do not approve the claim yourself.
Ground every factual claim in tool results or an explicitly applicable retrieved policy. A related topic is not evidence for the exact policy asked about (for example, coupon offers do not establish a student discount, and delivery slots do not establish midnight delivery). If the retrieved text does not directly establish the requested policy, say the information is unavailable and offer a human colleague. Do not invent policies, dates, amounts, order details, compensation, contact methods, or actions.
        For product Assured warranty questions, use the catalog for whether the SKU is Assured and warranty length, and the current warranty policy. Include the exact warranty duration and controlling claim process when applicable.
        If that policy says claims are raised in the app under My Orders, state those words clearly. Do not mention
price unless asked. If evidence is missing, say that clearly and offer a human colleague. Answer in at most two short sentences, avoid
repeating yourself, and use no more detail than the customer requested."""

    def _rewrite_query(self, message: str, conversation_query: str) -> str:
        """Normalize customer wording to the KB's section vocabulary without answering."""
        catalog = "\n".join(
            f"[{chunk.section_id}] {chunk.title} ({chunk.metadata.get('doc_type', 'unknown')}, "
            f"{chunk.metadata.get('status', 'unknown')})" for chunk in self.kb.chunks
        )
        prompt = f"""Rewrite the customer's request as a concise search query for the ZentroMart support knowledge base.
Use the KB's own vocabulary and section titles below (for example: return window, reverse pickup, refund initiation,
ZentroMart Assured, order status, identity verification, human handover). Translate Hindi/Hinglish intent into concise
English retrieval terms when useful, but preserve the customer's original meaning.
Keep all exact SKUs, order IDs, dates, amounts, product names, status words, and negation. Do not add facts, infer an
answer, or decide policy. Prefer terms matching current policy, then agent guides, help, FAQ, and marketing; historical
sections are not authoritative. Output only one search query, no explanation. The original customer message remains authoritative.

KB sections:
{catalog}

Recent conversation for context: {conversation_query}
Current customer message: {message}
Search query:"""
        try:
            rewritten = self.llm.chat([{"role": "user", "content": prompt}],
                                      temperature=0.0, num_predict=64).strip()
        except RuntimeError as exc:
            if self.verbose:
                print(f"[rewrite] unavailable; using original customer wording: {exc}", flush=True)
            return message
        rewritten = re.sub(r"^(?:search query|rewritten query)\s*:\s*", "", rewritten, flags=re.I).strip().strip('"')
        if not rewritten or len(rewritten) > max(500, len(message) * 4):
            return message
        if self.verbose:
            print(f"[rewrite] {rewritten}", flush=True)
        return rewritten

    def _llm_turn(self, message: str) -> str:
        language = _language_style(message)
        system_message = {"role": "system", "content": self._system_prompt(language)}
        if self.state.order_id and not self.state.verified_order:
            proof = self._verification_value(message)
            if proof:
                system_message["content"] += (
                    f"\nREQUIRED TOOL ACTION: Call lookup_order with order_id={self.state.order_id} "
                    f"and verification_value={proof} before answering.")
            else:
                system_message["content"] += (
                    f"\nREQUIRED TOOL ACTION: Call lookup_order with only order_id={self.state.order_id}. "
                    "Do not disclose order details; ask for verification after the tool result.")
        elif not self.state.order_id and _needs_order_id(message):
            system_message["content"] += (
                "\nREQUIRED DECISION: The customer asks about their specific order but has not supplied its ID. "
                "Set decision=clarify and ask for the order ID. Do not call lookup_order without an ID.")
        if _asks_for_human(message):
            system_message["content"] += (
                "\nREQUIRED TOOL ACTION: The customer explicitly requested a human. Call create_handover_ticket now; "
                "only confirm the handover if the tool returns created=true.")
        verified_standard_late_comp = (
            _is_late_delivery_compensation_request(message)
            and str((self.state.verified_order or {}).get("delivery_type", "")).casefold() == "standard")
        if verified_standard_late_comp:
            system_message["content"] += (
                "\nThe verified order is standard delivery. Follow the retrieved DEL-4/ESC-4 policy: "
                "decline discretionary late-delivery voucher/compensation requests; do not invent a voucher or credit.")
        compact_history = []
        for turn in self.state.conversation_history[-4:]:
            compact_history.extend([{"role": "user", "content": turn["user"]},
                                    {"role": "assistant", "content": turn["assistant"]}])
        # Keep full dialogue in state, but don't resend old KB chunks or tool payloads.
        self.state.messages = [system_message, *compact_history]
        self.state.current_issue = message
        product_query = " ".join([turn["user"] for turn in self.state.conversation_history[-4:]] + [message])
        rewritten_query = self._rewrite_query(message, product_query)
        retrieval_query = f"{product_query} {rewritten_query}"
        retrieval_started = time.perf_counter()
        chunks = self.kb.search(retrieval_query, limit=4)
        # The precedence rule is small, always relevant policy context and is explicitly
        # required by the capstone for resolving current-policy/FAQ conflicts.
        required_sections = ["KB-00.1"]
        if not self.state.conversation_history:
            required_sections.append("STY-1")
        normalized_query = retrieval_query.casefold()
        if _asks_for_human(message):
            required_sections.extend(("ESC-1", "ESC-2", "ESC-3"))
        if _is_late_delivery_compensation_request(message):
            required_sections.extend(("DEL-4", "ESC-4"))
        if re.search(r"\b(cannot deliver|can't deliver|could not deliver|unable to deliver|delivery attempt|"
                     r"failed delivery|address not found)\b", normalized_query):
            required_sections.append("DEL-3")
        is_shipped_cancellation = bool(
            re.search(r"\b(cancel|cancellation|cancelled)\b", normalized_query)
            and re.search(r"\b(ship|shipped|dispatch)\b", normalized_query))
        if is_shipped_cancellation:
            required_sections.extend(("CAN-2", "FAQH-1"))
        is_pickup_fee_question = bool(
            re.search(r"\b(fee|charged|charge|cost|free|pickup|collection)\b", normalized_query)
            and re.search(r"\b(pickup|collection|return|reverse pickup)\b", normalized_query))
        has_plus = bool(re.search(r"\bplus\b", normalized_query))
        damage_question = _is_damage_question(normalized_query)
        private_contact_request = _is_third_party_contact_request(normalized_query)
        unauthorized_activity = _is_unauthorized_activity(normalized_query)
        if is_pickup_fee_question:
            required_sections.append("RET-6")
            if has_plus:
                required_sections.append("PLS-2")
            elif damage_question:
                required_sections.append("DWM-2")
        if private_contact_request:
            required_sections.extend(("ESC-4", "IDV-4"))
        if unauthorized_activity:
            required_sections.extend(("ESC-1", "ESC-2", "ESC-3"))
        if re.search(r"\b(warranty|guarantee|assured)\b", normalized_query):
            required_sections.extend(("WAR-1", "WAR-2"))
        refund_timing = any(term in normalized_query for term in
                            ("refund", "refunds", "refund initiation", "refund initiated"))
        refund_start_date = self._delivery_date_from_text(product_query)
        if refund_timing:
            required_sections.append("REF-2")
            if refund_start_date:
                required_sections.append("KB-00.3")
        if damage_question:
            required_sections.extend(("DWM-1", "FAQH-3"))
        if self._is_return_window_question(retrieval_query) or re.search(r"\breturn\b", normalized_query):
            required_sections.extend(("RET-2", "RET-6"))
            delivered = self._delivery_date_from_text(product_query)
            if delivered and delivered < date(2026, 8, 1):
                required_sections.extend(("RET-3", "RETv3-2"))
        if re.search(r"\b(opened|used|seal(?:ed)?|unused|eligible|returnable|condition)\b|"
                     r"\b(can i return|still return|return this item)\b", normalized_query):
            required_sections.append("RET-4")
        for section_id in required_sections:
            required = next((chunk for chunk in self.kb.chunks if chunk.section_id == section_id), None)
            if required and all(chunk.section_id != section_id for chunk in chunks):
                chunks.append(required)
        self.state.evidence = [f"{chunk.source}:{chunk.section_id}" for chunk in chunks]
        self.state.retrieved_context = "\n\n".join(
            f"[{chunk.section_id}] {chunk.title}\n{chunk.text}" for chunk in chunks)
        if self.verbose:
            print(f"[rag] retrieved {len(chunks)} KB sections in {time.perf_counter() - retrieval_started:.1f}s: "
                  f"{', '.join(c.section_id for c in chunks)}", flush=True)
        context = self.state.retrieved_context or "No matching KB sections were found. Do not invent policy facts."
        if damage_question:
            for section_id in ("DWM-1", "FAQH-3"):
                chunk = next((item for item in self.kb.chunks if item.section_id == section_id), None)
                if chunk and f"{chunk.source}:{section_id}" not in self.state.evidence:
                    self.state.evidence.append(f"{chunk.source}:{section_id}")
                    self.state.retrieved_context += f"\n\n[{section_id}] {chunk.title}\n{chunk.text}"
        structured_context = []
        sku_match = re.search(r"\b[A-Z]{2,4}-\d{4}\b", product_query.upper())
        self.state.product_lookup_this_turn = False
        if sku_match:
            catalog_product = self.data.product(sku_match.group(0))
            products = [catalog_product] if catalog_product else []
        else:
            product_policy_fields = (("return_category",) if self._is_return_window_question(retrieval_query)
                                     else ("return_category", "assured", "warranty_months"))
            products = self.data.products_named_in(retrieval_query, stable_fields=product_policy_fields)
        product = products[0] if products else None
        if product:
            self.state.product_lookup_this_turn = True
            for row in products:
                sku = row["sku"]
                self.state.evidence.append(f"products.csv:{sku}")
            structured_context.append("Matching catalog records: " + json.dumps(products, ensure_ascii=False))
            if self._is_return_window_question(retrieval_query) and product.get("return_category") == "electronics":
                current_policy = next((chunk for chunk in self.kb.chunks if chunk.section_id == "RET-3"), None)
                if current_policy and f"{current_policy.source}:RET-3" not in self.state.evidence:
                    self.state.evidence.append(f"{current_policy.source}:RET-3")
                    self.state.retrieved_context += f"\n\n[RET-3] {current_policy.title}\n{current_policy.text}"
            if self._is_return_window_question(retrieval_query) and product.get("return_category") == "fashion":
                festive = next((chunk for chunk in self.kb.chunks if chunk.section_id == "RET-8"), None)
                if festive and f"{festive.source}:RET-8" not in self.state.evidence:
                    self.state.evidence.append(f"{festive.source}:RET-8")
                    self.state.retrieved_context += f"\n\n[RET-8] {festive.title}\n{festive.text}"
        if sku_match and not products:
            structured_context.append(f"No catalog record exists for SKU {sku_match.group(0)}; do not infer product facts.")
        high_value_damage = False
        if damage_question and product:
            try:
                high_value_damage = float(str(product.get("price_inr", "0")).replace(",", "")) > 10000
            except ValueError:
                high_value_damage = False
        if high_value_damage:
            for section_id in ("DWM-3", "ESC-1", "ESC-2", "ESC-3"):
                chunk = next((item for item in self.kb.chunks if item.section_id == section_id), None)
                if chunk and f"{chunk.source}:{section_id}" not in self.state.evidence:
                    self.state.evidence.append(f"{chunk.source}:{section_id}")
                    self.state.retrieved_context += f"\n\n[{section_id}] {chunk.title}\n{chunk.text}"
        if self.state.verified_order:
            order = self.state.verified_order
            safe_order = {key: order[key] for key in VERIFIED_ORDER_FIELDS if key in order}
            if order.get("items"):
                safe_order["items"] = order["items"]
            structured_context.append(f"Previously verified order facts: {json.dumps(safe_order, ensure_ascii=False)}")
        user_content = f"Customer message: {message}\n\nRetrieved KB context:\n{context}"
        if structured_context:
            user_content += "\n\nVerified structured data:\n" + "\n".join(structured_context)
        self.state.messages.append({"role": "user", "content": user_content})
        self.state.retrieved_evidence_this_turn = list(self.state.evidence)
        self.state.kb_searched_this_turn = False
        normalized_message = message.casefold()
        is_return_query = any(term in normalized_message for term in
                              ("return", "वापस", "लौट", "रिटर्न"))
        explicitly_business_days = bool(re.search(
            r"\b(business|working)\s+days?\b|कार्य\s*दिवस|कारोबारी\s*दिन", normalized_message))
        needs_refund_timing = any(term in normalized_message for term in
                                  ("refund", "रिफंड", "पैसे वापस"))
        allow_date_tool = explicitly_business_days or needs_refund_timing or not is_return_query
        available_tools = [tool for tool in TOOLS
                           if (tool["function"]["name"] != "lookup_order" or self.state.order_id)
                           and (tool["function"]["name"] != "calculate_business_day_range" or allow_date_tool)]
        max_rounds = 3 if available_tools else 1
        turn_tool_results = []
        for round_number in range(max_rounds):
            if self.verbose:
                print(f"[llm] response/tool decision (round {round_number + 1}/{max_rounds})", flush=True)
            llm_started = time.perf_counter()
            self.state.llm_called_this_turn = True
            response = self.llm.chat_response(self.state.messages, tools=available_tools or None, temperature=0.1)
            if self.verbose:
                print(f"[llm] call completed in {time.perf_counter() - llm_started:.1f}s", flush=True)
            assistant_message = response.get("message", {})
            self.state.messages.append(assistant_message)
            calls = assistant_message.get("tool_calls") or []
            if not calls:
                reply = (assistant_message.get("content") or "I couldn't prepare a response.").strip()
                if reply.casefold() in {"i couldn't prepare a response.", "i’m sorry, i couldn’t prepare a response.",
                                        "i'm having trouble completing that request."}:
                    reply = self._unknown_reply(language)
                date_result = next((result for name, result in reversed(turn_tool_results)
                                    if name == "calculate_business_day_range" and result.get("maximum_date")), None)
                if date_result and date_result["maximum_date"] not in reply:
                    reply += f" The full processing window is {date_result['minimum_date']} to {date_result['maximum_date']}."
                return reply
                return (assistant_message.get("content") or "I’m sorry, I couldn’t prepare a response.").strip()
            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                args = function.get("arguments", {})
                if self.verbose:
                    print(f"[tool] {name}({json.dumps(args, ensure_ascii=False)})", flush=True)
                result = self._call_tool(name, args, message)
                turn_tool_results.append((name, result))
                self.state.previous_tool_results.append({"name": name, "result": result})
                self.state.previous_tool_results = self.state.previous_tool_results[-6:]
                self.state.tool_results_this_turn.append({"name": name, "result": result})
                if self.verbose:
                    brief = json.dumps(result, ensure_ascii=False)
                    print(f"[tool result] {name}: {brief[:500]}", flush=True)
                self.state.messages.append({"role": "tool", "tool_name": name,
                                            "content": json.dumps(result, ensure_ascii=False)})
        return "I’m having trouble completing that request. I can connect you with a human support colleague."

    def _unavailable_reply(self, message: str) -> str:
        # Never paste a raw, possibly irrelevant KB chunk as a customer-facing answer.
        # If generation is unavailable, be transparent and offer a human colleague.
        self.state.evidence = []
        self.state.decision = "answer"
        return self._unknown_reply(_language_style(message))

    def _translate_reply(self, reply: str, language: str) -> str:
        instruction = {
            "hi": "Rewrite this answer in natural, polite Hindi using Devanagari script.",
            "hi-en": "Rewrite this answer in natural conversational Hinglish using Latin script.",
        }[language]
        messages = [
            {"role": "system", "content": (
                instruction + " Preserve every factual detail, date, amount, condition, and action exactly. "
                "Do not add facts, remove qualifications, change the decision, or call tools. Return only the rewritten answer.")},
            {"role": "user", "content": reply},
        ]
        return self.llm.chat(messages, temperature=0.0, num_predict=120).strip()

    def respond(self, message: str) -> dict:
        self.state.tool_calls = []
        self.state.decision = "answer"
        self.state.kb_searched_this_turn = False
        self.state.product_lookup_this_turn = False
        self.state.ticket_created_this_turn = False
        self.state.llm_called_this_turn = False
        self.state.tool_results_this_turn = []
        self.state.retrieved_evidence_this_turn = []
        llm_error = None
        self._remember_order_id(message)
        if _is_third_party_contact_request(message):
            self.state.decision = "refuse"
        elif not self.state.order_id and _needs_order_id(message):
            self.state.decision = "clarify"
        if (_is_refund_eta_question(message) and not _has_payment_method(message)
                and not self.state.verified_order):
            self.state.decision = "clarify"
        if (_is_late_delivery_compensation_request(message)
                and str((self.state.verified_order or {}).get("delivery_type", "")).casefold() == "standard"):
            self.state.decision = "refuse"
        try:
            reply = self._llm_turn(message) if self.use_llm else self._unavailable_reply(message)
        except RuntimeError as exc:
            llm_error = str(exc)
            if self.verbose:
                print(f"[llm error] {llm_error}", flush=True)
            reply = self._unavailable_reply(message)
        self.state.retrieved_evidence_this_turn = list(self.state.evidence)
        if not self.state.ticket_created and re.search(r"(?i)(ticket.{0,30}(created|raised|generated)|(?:created|raised|generated).{0,30}ticket)", reply):
            reply = "I can offer a human handover, but I cannot confirm that a support ticket was created."
        language = _language_style(message)
        if language == "hi" and not re.search(r"[\u0900-\u097f]{3,}", reply):
            if self.verbose:
                print("[language] correcting response to Hindi", flush=True)
            try:
                translated = self._translate_reply(reply, language) if self.llm else ""
            except RuntimeError as exc:
                llm_error = llm_error or str(exc)
                translated = ""
            if re.search(r"[\u0900-\u097f]{3,}", translated):
                reply = translated
            else:
                reply = self._unknown_reply("hi")
                self.state.evidence = []
        elif language == "hi-en" and not (set(re.findall(r"[a-z]+", reply.casefold())) & _HINGLISH_MARKERS):
            if self.verbose:
                print("[language] correcting response to Hinglish", flush=True)
            try:
                translated = self._translate_reply(reply, language) if self.llm else ""
            except RuntimeError as exc:
                llm_error = llm_error or str(exc)
                translated = ""
            translated_tokens = set(re.findall(r"[a-z]+", translated.casefold()))
            if translated_tokens & _HINGLISH_MARKERS:
                reply = translated
            else:
                reply = self._unknown_reply("hi-en")
                self.state.evidence = []
        self.state.conversation_history.append({"user": message, "assistant": reply})
        return {"reply": reply, "sources": list(self.state.evidence),
                "retrieved_evidence": list(self.state.retrieved_evidence_this_turn),
                "order_verified": bool(self.state.verified_order),
                "decision": self.state.decision, "tool_calls": list(self.state.tool_calls),
                "ticket_created": self.state.ticket_created,
                "ticket_created_this_turn": self.state.ticket_created_this_turn,
                "tool_results": list(self.state.tool_results_this_turn),
                "llm_error": llm_error,
                "llm_used": bool(self.use_llm and llm_error is None and self.state.llm_called_this_turn)}

    def handover(self, reason: str) -> dict:
        from tools.tools import create_ticket

        result = create_ticket(f"Customer needs human support. Reason: {reason}",
                               order_id=self.state.order_id, reason="agent_handover")
        self.state.ticket_created = bool(result.get("created"))
        return result
