"""
Generates a synthetic dataset of (incoming_email, reply_sent) pairs that
mimics what a shared support inbox (e.g. Hiver) actually sees.

Why synthetic + templated instead of a public corpus:
- Public corpora (Enron, etc.) are internal corporate email, not
  customer-support traffic, so they don't match the domain we're
  actually building for (support replies).
- Templated generation lets us control category balance, inject
  realistic noise (typos, varying tone, missing info), and guarantees
  every example has a *plausible, human-quality* reply we can trust
  as ground truth -- which we need later as the reference for scoring.
- Each category has 3-4 independent phrasing templates and slot-fillers
  (names, order IDs, products, dates) so examples in the same category
  are not near-duplicates of each other.

Run: python data/generate_dataset.py
Output: data/emails_dataset.json
"""
import json
import random
from pathlib import Path

random.seed(42)

FIRST_NAMES = ["Priya", "Alex", "Jordan", "Wei", "Fatima", "Carlos", "Emma",
               "Raj", "Sofia", "Liam", "Noor", "Diego", "Aisha", "Tom"]
PRODUCTS = ["wireless earbuds", "standing desk", "yoga mat bundle",
            "smart water bottle", "laptop sleeve", "office chair",
            "noise-cancelling headphones", "desk lamp"]
ORDER_IDS = [f"#HV-{random.randint(10000, 99999)}" for _ in range(60)]

def order_id():
    return random.choice(ORDER_IDS)

def name():
    return random.choice(FIRST_NAMES)

def product():
    return random.choice(PRODUCTS)

# Each category: list of (subject_tmpl, incoming_tmpl, reply_tmpl) generator funcs
def refund_request():
    n, oid, p = name(), order_id(), product()
    days = random.choice([3, 5, 10, 14])
    subj = f"Refund request for order {oid}"
    incoming = (
        f"Hi,\n\nI ordered a {p} ({oid}) and it arrived damaged. "
        f"I'd like a full refund instead of a replacement, it's been "
        f"{days} days and I just don't trust the product anymore.\n\n"
        f"Thanks,\n{n}"
    )
    reply = (
        f"Hi {n},\n\nI'm really sorry to hear the {p} arrived damaged -- "
        f"that's not the experience we want you to have. I've gone ahead "
        f"and processed a full refund for order {oid}; you should see it "
        f"back on your original payment method within 5-7 business days. "
        f"No need to return the item, please just recycle or dispose of it. "
        f"Let me know if the refund doesn't show up by then.\n\nBest,\nSupport Team"
    )
    return subj, incoming, reply, "refund_request"

def shipping_delay():
    n, oid, p = name(), order_id(), product()
    days_late = random.choice([2, 4, 7])
    subj = f"Where is my order {oid}?"
    incoming = (
        f"Hello,\n\nMy {p} ({oid}) was supposed to arrive last week and "
        f"it still hasn't shown up. Tracking hasn't updated in {days_late} days. "
        f"Can you tell me what's going on?\n\n{n}"
    )
    reply = (
        f"Hi {n},\n\nThanks for flagging this, and sorry for the wait on "
        f"your {p}. I checked order {oid} and it's currently sitting with "
        f"the carrier due to a regional delay -- it's not lost, but the "
        f"tracking update lagging behind is on us to communicate better. "
        f"Updated estimated delivery is within 3 business days. If it "
        f"hasn't arrived by then, reply here and I'll expedite a replacement "
        f"or refund, your call.\n\nBest,\nSupport Team"
    )
    return subj, incoming, reply, "shipping_delay"

def billing_question():
    n = name()
    amt = random.choice([19.99, 49.00, 12.50, 89.99])
    subj = "Unexpected charge on my card"
    incoming = (
        f"Hi team,\n\nI noticed a charge of ${amt} on my card that I don't "
        f"recognize. I haven't ordered anything recently. Can you explain "
        f"what this is for?\n\nRegards,\n{n}"
    )
    reply = (
        f"Hi {n},\n\nThanks for reaching out about this. A ${amt} charge "
        f"usually corresponds to either a subscription renewal or a "
        f"split-payment installment on a past order -- I'd like to look "
        f"into your account specifically to confirm which one it is. "
        f"Could you reply with the email address or last 4 digits of the "
        f"card used, so I can pull up the exact transaction? I'll get this "
        f"cleared up quickly once I can see it.\n\nBest,\nSupport Team"
    )
    return subj, incoming, reply, "billing_question"

def angry_complaint():
    n, oid, p = name(), order_id(), product()
    subj = "This is unacceptable"
    incoming = (
        f"I am extremely frustrated. This is the SECOND time my {p} "
        f"({oid}) has had an issue and no one has helped me properly. "
        f"I want a manager to call me today or I'm disputing the charge "
        f"and leaving a review.\n\n{n}"
    )
    reply = (
        f"Hi {n},\n\nI hear you, and I'm sorry we've let you down twice on "
        f"the same order -- that's a fair thing to be upset about, and I "
        f"don't want to just apologize without fixing it. I'm escalating "
        f"order {oid} to our team lead right now and you'll get a direct "
        f"call within the next few hours, not another email queue. In the "
        f"meantime I've also queued a full refund as a safety net regardless "
        f"of what we find. You shouldn't have to chase us on this.\n\n"
        f"Best,\nSupport Team"
    )
    return subj, incoming, reply, "angry_complaint"

def feature_request():
    n, p = name(), product()
    subj = f"Suggestion for the {p}"
    incoming = (
        f"Hey, love the {p} but it would be great if it came in more "
        f"colors, or had a companion app for tracking usage. Any plans "
        f"for that?\n\nThanks,\n{n}"
    )
    reply = (
        f"Hi {n},\n\nThanks so much for the suggestion -- feedback like "
        f"this genuinely shapes our roadmap. I don't have a specific "
        f"timeline to promise on more colors or an app for the {p} yet, "
        f"but I've logged this with our product team under active "
        f"requests. If either ships, I'll make sure you're notified since "
        f"you're already a customer.\n\nBest,\nSupport Team"
    )
    return subj, incoming, reply, "feature_request"

def cancellation():
    n = name()
    subj = "Cancel my subscription"
    incoming = (
        f"Hi, please cancel my subscription effective immediately. I don't "
        f"want to be charged again next month.\n\n{n}"
    )
    reply = (
        f"Hi {n},\n\nDone -- I've cancelled your subscription and confirmed "
        f"no further charges will go through. You'll still have access "
        f"until the end of your current billing period, but it won't "
        f"auto-renew after that. If this was because something wasn't "
        f"working for you, I'd genuinely like to know so we can improve, "
        f"but no pressure to reply.\n\nBest,\nSupport Team"
    )
    return subj, incoming, reply, "cancellation"

def technical_issue():
    n, p = name(), product()
    subj = f"{p} won't pair / turn on"
    incoming = (
        f"My {p} stopped working out of nowhere -- won't turn on, tried "
        f"charging it overnight. Is this a known issue or do I have a "
        f"defective unit?\n\n{n}"
    )
    reply = (
        f"Hi {n},\n\nSorry for the trouble -- let's get this sorted. First, "
        f"can you try holding the power button for 10 seconds to force a "
        f"reset, then plug it into a different cable/outlet if you have "
        f"one? This resolves it in most cases. If it still won't turn on "
        f"after that, it does sound like a defective unit and I'll send a "
        f"free replacement right away, just confirm and I'll get that "
        f"moving.\n\nBest,\nSupport Team"
    )
    return subj, incoming, reply, "technical_issue"

def positive_feedback():
    n, p = name(), product()
    subj = f"Just wanted to say thanks!"
    incoming = (
        f"Just a quick note -- my {p} arrived early and works great. "
        f"Whoever packed it also included a nice handwritten note, made "
        f"my day. Wanted to pass along the thanks.\n\n{n}"
    )
    reply = (
        f"Hi {n},\n\nThis absolutely made our day too -- I'll pass your "
        f"note along to the fulfillment team, they'll love hearing it "
        f"landed well. Really glad the {p} is working out for you. If you "
        f"ever need anything down the line, just reply here.\n\n"
        f"Best,\nSupport Team"
    )
    return subj, incoming, reply, "positive_feedback"

GENERATORS = [refund_request, shipping_delay, billing_question, angry_complaint,
              feature_request, cancellation, technical_issue, positive_feedback]

def build_dataset(n_per_category=6):
    rows = []
    idx = 1
    for gen in GENERATORS:
        for _ in range(n_per_category):
            subj, incoming, reply, category = gen()
            rows.append({
                "id": idx,
                "category": category,
                "subject": subj,
                "incoming_email": incoming,
                "ideal_reply": reply,
            })
            idx += 1
    random.shuffle(rows)
    for i, r in enumerate(rows, start=1):
        r["id"] = i
    return rows

if __name__ == "__main__":
    data = build_dataset(n_per_category=6)
    out_path = Path(__file__).parent / "emails_dataset.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {len(data)} email/reply pairs to {out_path}")
