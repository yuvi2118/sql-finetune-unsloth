SYSTEM = (
    "You are a SQL expert. Given a database schema and a question, "
    "reply with a single SQL query and nothing else."
)


def build_messages(context, question):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{context}\n\nQuestion: {question}"},
    ]
