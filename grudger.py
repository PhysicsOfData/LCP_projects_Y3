team_name = 'A2'
strategy_name = 'Grudger'
strategy_description = 'Starts with cooperation and once if the opponent betrays then always grudger betrays irrespective of the opponent move.'
    
def move(my_history, their_history, my_score, their_score):
    if(len(my_history) == 0 and len(their_history) == 0):
        return 'c'
    if(my_history[-1] == 'c' and their_history[-1] == 'c'):
        return 'c'
    if(my_history[-1] == 'b' or their_history[-1] == 'b'):
        return 'b'
    
