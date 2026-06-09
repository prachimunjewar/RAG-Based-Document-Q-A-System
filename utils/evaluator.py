from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def score_answer(question, answer, context_chunks):
    # Faithfulness — how grounded is the answer in the context
    context_text = " ".join(context_chunks)
    faith_score = float(util.cos_sim(
        model.encode(answer),
        model.encode(context_text)
    )[0][0])

    # Relevancy — how well does the answer match the question
    rel_score = float(util.cos_sim(
        model.encode(answer),
        model.encode(question)
    )[0][0])

    return {
        "faithfulness": round(faith_score * 100, 1),
        "answer_relevancy": round(rel_score * 100, 1)
    }
