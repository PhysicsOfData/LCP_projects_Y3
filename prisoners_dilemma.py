import random

def main_play(modules):
    scores, moves = play_tournament(modules)
    # section0, section1, section2, section3 = make_reports(modules, scores, moves)
    # print(section0+section1+section2)
    # post_to_file(section0+section1+section2 )
    return scores, moves
        
def play_tournament(modules):

    zeros_list = [0]*len(modules) 
    scores = [zeros_list[:] for module in modules] 
    moves = [zeros_list[:] for module in modules] 
    for first_team_index in range(len(modules)):
        for second_team_index in range(first_team_index):
            player1 = modules[first_team_index]
            player2 = modules[second_team_index]
            score1, score2, moves1, moves2 = play_iterative_rounds(player1, player2)
            scores[first_team_index][second_team_index] = score1/len(moves1) 
            moves[first_team_index][second_team_index] = moves1
            scores[second_team_index][first_team_index] = score2/len(moves2) 
            moves[second_team_index][first_team_index] = moves2

        scores[first_team_index][first_team_index] = 0
        moves[first_team_index][first_team_index] = ''
    return scores, moves


def play_iterative_rounds(player1, player2):

    number_of_rounds = random.randint(100, 200)
    moves1 = ''
    moves2 = ''
    score1 = 0
    score2 = 0
    for round in range(number_of_rounds):
        score1, score2, moves1, moves2 = play_round(player1, player2, score1, score2, moves1, moves2)
    return (score1, score2, moves1, moves2)
    
def play_round(player1, player2, score1, score2, moves1, moves2):
    
    RELEASE = 250 
    TREAT = 400 
    SEVERE_PUNISHMENT = 0 
    PUNISHMENT = 100 
    
    # Keep T > R > P > S 
    # Keep 2R > T + S 
    
    ERROR = -250
    
    action1 = player1.move(moves1, moves2, score1, score2)
    action2 = player2.move(moves2, moves1, score2, score1)
    if (type(action1) != str) or (len(action1) != 1):
        action1=' '
    if (type(action2) != str) or (len(action2) != 1):
        action2=' '
    
    actions = action1 + action2
    if actions == 'cc':
        score1 += RELEASE
        score2 += RELEASE
    elif actions == 'cb':
        score1 += SEVERE_PUNISHMENT
        score2 += TREAT
    elif actions == 'bc':
        score1 += TREAT
        score2 += SEVERE_PUNISHMENT 
    elif actions == 'bb':   
        score1 += PUNISHMENT
        score2 += PUNISHMENT     
    else:
        score1 += ERROR
        score2 += ERROR
    
    if action1 in 'bc':
        moves1 += action1
    else:
        moves1 += ' '
    if action2 in 'bc':
        moves2 += action2
    else:
        moves2 += ' '
                    
    return (score1, score2, moves1, moves2)

def make_reports(modules, scores, moves):
    section0 = make_section0(modules, scores)
    section1 = make_section1(modules, scores)
    section2 = make_section2(modules, scores)
    return section0, section1, section2
        
def make_section0(modules, scores):

    section0 = '-'*80+'\n'
    section0 += 'Section 0 - Line up\n'
    section0 += '-'*80+'\n'
    for index in range(len(modules)):
        section0 += 'Player ' + str(index) + ' (P' + str(index) + '): '
        section0 +=  str(modules[index].strategy_name) + '\n'
        strategy_description = str(modules[index].strategy_description)
        while len(strategy_description) > 1:
            newline = strategy_description[:72].find('\n')
            if newline> -1:
                section0 += ' '*8 + strategy_description[:newline+1]
                strategy_description = strategy_description[newline+1:]
            else:
                section0 += ' '*8 + strategy_description[:72] + '\n'
                strategy_description = strategy_description[72:]
    return section0
    
def make_section1(modules, scores):

    section1 = '-'*80+'\nSection 1 - Player vs. Player\n'+'-'*80+'\n'
    section1 += 'Each column shows pts/round earned against each other player:\n'
    section1 += '        '
    for i in range(len(modules)):
          section1 += '{:>7}'.format('P'+str(i))
    section1 += '\n'
    for index in range(len(modules)):
        section1 += 'vs. P' + str(index) + ' :'
        for i in range(len(modules)):
            section1 += '{:>7}'.format(scores[i][index])
        section1 += '\n'

    section1 += 'TOTAL  :'
    for index in range(len(modules)):
        section1 += '{:>7}'.format(sum(scores[index]))     
    return section1+'\n'
    
def make_section2(modules, scores):

    section2 = '-'*80+'\nSection 2 - Leaderboard\n'+'-'*80+'\n'
    section2 += 'Average points per round:\n'
    section2 += '(P#):  Score      with strategy name\n'
    
    section2_list = []
    for index in range(len(modules)):
        section2_list.append((
                              'P'+str(index),
                              str(sum(scores[index])/len(modules)),
                              str(modules[index].strategy_name)))
    section2_list.sort(key=lambda x: int(x[1]), reverse=True)
    
    for team in section2_list:
        Pn, n_points, strategy_name = team
        section2 += '({}): {:<10} points with {:<40}\n'.format( Pn, n_points, strategy_name[:40])                       
    return section2 
