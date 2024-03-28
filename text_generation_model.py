from transformers import pipeline


def query_generation_model(question: str, context: str, model_name_or_path: str):
    question_answering_pipeline = pipeline('question-answering', model=model_name_or_path, tokenizer=model_name_or_path)
    model_input = {'question': question, 'context': context}
    res = question_answering_pipeline(model_input)
    return res['answer']
