# Writing style

Rules for prose we write in this repo: the paper (`paper/DRAFT.md`), the docs, commit
messages, and any narrative text. The goal is writing that reads as human-written and is
easy to read. This is about voice and readability, not about hiding authorship (our AI
contribution is acknowledged separately and openly).

Distilled from the Wikipedia essay "Signs of AI writing"
(https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing). When in doubt, write the way
a careful researcher writes to a colleague: plain, specific, and direct.

## 1. Punctuation and typography

- No em dashes. This is the single most common tell. Replace `—` with a comma, a colon,
  parentheses, or two sentences. "The result is clear, VF is detected" or "The result is
  clear: VF is detected", not "The result is clear — VF is detected".
- En dashes are fine for numeric ranges only: `3–5 Hz`, `Q1–Q8`.
- Straight quotes and apostrophes (`"` `'`), never curly ones (`"` `'`).
- Headings in sentence case, not Title Case. "Signal preprocessing", not "Signal
  Preprocessing".
- Do not put a horizontal rule (`---`) right before a heading, and do not skip heading
  levels (H2 to H4).

## 2. Words and phrases to avoid

Overused "AI vocabulary". Prefer the plain word on the right.

- delve into → examine, look at
- underscore, highlight, showcase → show
- leverage, utilize → use
- boasts → has
- foster → support, encourage
- pivotal, crucial → key, important (and use sparingly)
- meticulous, intricate, robust → drop or say what you actually mean
- align with, resonate with → match, agree with
- garner → get, receive
- Stock metaphors: tapestry, landscape (abstract), realm, navigate (abstract), seamless.
- Puffery: vibrant, nestled, rich heritage, groundbreaking, renowned, in the heart of.

Empty filler phrases to cut: "it is important to note", "it is worth noting", "plays a
significant/vital role", "a wide range of", "in today's world".

## 3. Sentence and paragraph structure

- No negative parallelism. Avoid "not only X but also Y", "it's not X, it's Y", "X rather
  than Y" as a rhetorical frame. State the point directly.
- Avoid the rule of three. Three adjectives or three clauses in a row create false rhythm
  and false completeness. Use the number of items the content actually needs.
- Do not dodge "is" and "are". "TCSC is a time-domain measure" beats "TCSC serves as / acts
  as / represents a time-domain measure".
- Do not force synonyms (elegant variation). Repeat the key term. In a paper, calling the
  same thing "the algorithm", "the method", and "the approach" in one paragraph confuses
  more than it varies.
- Cut inflated significance and trailing "-ing" claims. "...N_d = 48, establishing it as a
  key benchmark" adds nothing. State the fact and stop.
- Vary sentence length. If every sentence has the same shape and length, it reads
  mechanically.

## 4. Formatting

- Bold sparingly: the first definition of a term, or genuine emphasis. Do not bold every
  key term, and do not write bullet lists where each item starts with a **bolded lead:**.
- Prefer prose to bullet lists for arguments and explanations. Use lists only for things
  that are genuinely a list (databases, parameters, steps).
- No emoji as decoration or structure in the paper or docs.

## 5. Openings, conclusions, and tone

- No recap sentences opening with "In summary", "In conclusion", or "Overall", unless the
  document actually has a conclusion section that earns it.
- No template endings like "Despite its X, it faces challenges...".
- Neutral and specific, not promotional. Give the fact or number; let it speak.

## 6. Hedging and attribution

- No weasel attributions: "studies suggest", "it is widely believed", "observers note".
  Name the source and cite it, or drop the claim.
- Do not hedge every sentence. Commit to the claim, or state the real uncertainty.

## 7. Integrity (matters most for the paper)

- Never invent citations, DOIs, numbers, or results. Every reference must be a real paper
  from `docs/papers/` or `docs/texts/`, and every reported figure must trace to a source or
  to our own experiments.

## Self-check before saving prose

Before committing prose, scan for: `—` (em dash), curly quotes, the words in section 2,
"not only", "it is important to note", three-in-a-row adjectives, bulleted **bold leads**,
and "In summary". Fix them.
