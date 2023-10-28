def Selection(depression_type):
    
    depression_type = 'depression'
    activities = {
        'meditate': {'time': 'day', 'param2': 'meditate', 'text': 'Meditate'},#
        'deep_breathing': {'time': 'night', 'param2': 'deep_breathing', 'text': 'Deep Breathing'},#
        'go_for_a_walk': {'time': 15, 'param2': 'go_for_a_walk', 'text': 'Go for a Walk'},#
        'listen_to_music': {'time': 15, 'param2': 'listen_to_music', 'text': 'Listen to Music'}, #re-direct to pura's component
        'do_yoga': {'time': 15, 'param2': 'do_yoga', 'text': 'Do Yoga'},#
        'write_journal': {'time': 15, 'param2': 'write_journal', 'text': 'Write in a Journal'},#
        'take_a_nap': {'time': 15, 'param2': 'take_a_nap', 'text': 'Take a Nap'},
        'call_a_friend': {'time': 15, 'param2': 'call_a_friend', 'text': 'Call a Friend'},#
        'read_a_book': {'time': 15, 'param2': 'read_a_book', 'text': 'Read a Book'},#
        'cook_a_nice_meal': {'time': 15, 'param2': 'cook_a_nice_meal', 'text': 'Cook a Nice Meal'},
        'gardening': {'time': 15, 'param2': 'gardening', 'text': 'Do Gardening'},#
        'play_with_pets': {'time': 15, 'param2': 'play_with_pets', 'text': 'Play with Pets'},
        'watch_funny_videos': {'time': 15, 'param2': 'watch_funny_videos', 'text': 'Watch Funny Videos'},
        'do_something_creative': {'time': 15, 'param2': 'do_something_creative', 'text': 'Do Something Creative'},
        'do_exercise': {'time': 15, 'param2': 'do_exercise', 'text': 'Do Exercise'},#
        'take_a_hot_bath': {'time': 15, 'param2': 'take_a_hot_bath', 'text': 'Take a Hot Bath'},
        'practice_mindfulness': {'time': 15, 'param2': 'practice_mindfulness', 'text': 'Practice Mindfulness'},
        'solve_puzzles': {'time': 15, 'param2': 'solve_puzzles', 'text': 'Solve Puzzles'},
        'volunteer_for_a_cause': {'time': 15, 'param2': 'volunteer_for_a_cause', 'text': 'Volunteer for a Cause'},
       
    }

    # Function to select activities based on the type of depression
    def select_activities(depression_type):
        # Define different activity recommendations for each depression type
        depression_activities = {
            'depression' : ['go_for_a_walk','do_yoga','gardening','do_exercise','call_a_friend','meditate'],
            'anxiety' : ['deep_breathing', 'do_yoga','meditate','read_a_book', 'take_a_hot_bath'],
            'bipolar' : ['write_journal', 'read_a_book', 'watch_funny_videos','deep_breathing','call_a_friend'],
            'bpd' : ['meditate','deep_breathing','listen_to_music','take_a_hot_bath','do_exercise','write_journal'],
            'schizophrenia' : ['solve_puzzles','volunteer_for_a_cause','do_exercise','deep_breathing','write_journal'],
            'autism' : ['volunteer_for_a_cause','gardening','cook_a_nice_meal','do_exercise','solve_puzzles'],
            'stress' : ['meditate','deep_breathing','call_a_friend','do_yoga','take_a_hot_bath','take_a_nap','listen_to_music']
        }

        return [activities[activity]['text'] for activity in depression_activities.get(depression_type, [])]

    recommended_activities = select_activities(depression_type)

    return recommended_activities
