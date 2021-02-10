####
# Each team's file must define four tokens:
#     team_name: a string
#     strategy_name: a string
#     strategy_description: a string
#     move: A function that returns 'c' or 'b'
####

team_name = 'A3'
strategy_name = 'Tit for Tat'
strategy_description = 'Tit for tat.'
    
def move(my_history, their_history, my_score, their_score):
    if len(my_history)==0: 
        return 'c'
    else:
        return their_history[-1]