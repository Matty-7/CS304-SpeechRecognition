# hmm_recognition.py

def recognize_speech(hmms, test_data):
    best_match = None
    highest_score = float('-inf')

    for hmm in hmms:
        score = hmm.calculate_score(test_data)
        if score > highest_score:
            highest_score = score
            best_match = hmm

    return best_match
