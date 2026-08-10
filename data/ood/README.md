# Out of domain test set

Use `customer_feedback_template.csv` to record human labeled customer feedback style messages.

Rules:

1. Use only labels listed in `src/goemotions_project/labels.py`.
2. Separate multiple labels with a vertical bar, for example `confusion|annoyance`.
3. Have at least two team members label each message independently when possible.
4. Resolve disagreements before the final evaluation.
5. Do not use these examples for training, threshold selection, or hyperparameter tuning.
6. Do not include private or personally identifying customer information.

