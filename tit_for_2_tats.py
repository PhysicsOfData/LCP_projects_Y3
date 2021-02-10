team_name = 'A3'
strategy_name = 'Tit for 2 Tats'
strategy_description = 'Tit for 2 tats.'
    
def move(my_history, their_history, my_score, their_score):
    if len(my_history)==0: 
        return 'c'
    x = len(their_history)
    if(x >= 2):
        if(their_history[x-1] == 'b' and their_history[x-2] == 'b'):
            return 'b'
        else:
            return 'c'
    else:
        return 'c'