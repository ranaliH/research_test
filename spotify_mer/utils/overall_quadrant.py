from collections import Counter

def categorize_emotion(number):
    if number == 1:
        return "Happy"
    elif number == 2:
        return "Angry"
    elif number == 3:
        return "Tender"
    elif number == 4:
        return "Sad"


def get_overall_quadrant(quadrants):

    counter = Counter(quadrants)
    most_common = counter.most_common(1)

    most_common_number = most_common[0][0]
    emotion = categorize_emotion(most_common_number)
    # frequency = most_common[0][1]
    
    print("Current emotion based on recently played songs:", emotion)
    # print("Frequency:", frequency)
    return emotion