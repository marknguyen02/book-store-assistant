from tasks.intent_classifier import classify_task
from tasks.lookup import lookup_book_data
from tasks.recommend import get_recommendation
from tasks.order import handle_order
from tasks.fallback import handle_fallback

def handle_user_request(request, history):
    task = classify_task(request)

    if task == "lookup":
        result = lookup_book_data(request)
    elif task == "recommend":
        result = get_recommendation(request)
    elif task == "order":
        result = handle_order(request)
    else:
        result = handle_fallback(request, history)

    return result