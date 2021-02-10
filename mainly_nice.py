####
# Each team's file must define four tokens:
#     team_name: a string
#     strategy_name: a string
#     strategy_description: a string
#     move: A function that returns 'c' or 'b'
####

team_name = 'A4'
strategy_name = 'Mainly nice'
strategy_description = 'Mainly nice.'
    
def move(my_history, their_history, my_score, their_score):

    if len(my_history) < 5:
        return 'c'
    else:
        if len(my_history) % 2 == 0:
            return 'c'
        else:
            return 'b'
