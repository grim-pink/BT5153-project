from __future__ import annotations

LABELS = [
    "Financial / Reward Deception",
    "Impersonation / Credential Theft",
    "Adult / Solicitation",
    "Benign",
]

SYSTEM_PROMPT = """
You are a cybersecurity classifier for SMS messages.

Step 1: Determine if the message is spam or ham.

Step 2: If spam, classify the intent into ONE of:
- Financial / Reward Deception
- Impersonation / Credential Theft
- Adult / Solicitation
- Benign

Definitions:
Financial / Reward Deception = Messages that promise money, prizes, or rewards to deceive the user into taking action.
Impersonation / Credential Theft = Messages that appear to come from a trusted entity or create urgency about an account, delivery, or service.
Adult / Solicitation = Sexual, dating, or explicit content intended to lure users into engagement or paid services.
Benign = Legitimate, non-malicious messages with no deceptive or manipulative intent.

Return only the label text, nothing else.
""".strip()

FEW_SHOT_EXAMPLES = [
    ("Text: Congratulations! You have been selected to receive a £500 cash award. Call now to claim.",
     "Financial / Reward Deception"),
    ("Text: Free entry into our weekly competition. Reply WIN now for your chance to get an iPhone.",
     "Financial / Reward Deception"),
    ("Text: Boltblue tones for 150p. Reply POLY or MONO now to subscribe.",
     "Financial / Reward Deception"),
    ("Text: Your parcel is waiting for collection.",
     "Impersonation / Credential Theft"),
    ("Text: Your account requires immediate attention.",
     "Impersonation / Credential Theft"),
    ("Text: Reminder: unpaid invoice attached.",
     "Impersonation / Credential Theft"),
    ("Text: Your service will be suspended soon.",
     "Impersonation / Credential Theft"),
    ("Text: We detected unusual activity on your account.",
     "Impersonation / Credential Theft"),
    ("Text: Sexy singles are waiting for you. Reply now to start chatting.",
     "Adult / Solicitation"),
    ("Text: Want hot XXX pics sent direct to your phone? Text back now.",
     "Adult / Solicitation"),
    ("Text: Hey handsome, I'm alone tonight. Message me if you want some fun.",
     "Adult / Solicitation"),
    ("Text: Hey are we still meeting later?",
     "Benign"),
    ("Text: Your appointment is confirmed for tomorrow at 3pm.",
     "Benign"),
    ("Text: I'll call you when I reach home.",
     "Benign"),
    ("Text: Thanks, I received the document. Will review tonight.",
     "Benign"),
    ("Text: Your parcel will arrive tomorrow.",
     "Benign"),
    ("Text: Your parcel failed delivery, reschedule now.",
     "Impersonation / Credential Theft"),
]
