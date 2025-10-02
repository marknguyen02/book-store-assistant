from tasks import (
    classify_task,
    lookup_book_data,
    get_recommendation,
    handle_order,
    handle_fallback
)

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