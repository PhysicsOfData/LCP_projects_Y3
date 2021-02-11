import math
import random
import os.path
import prisoners_dilemma
import matplotlib.pyplot as plt

import nice_guy, bad_guy, mainly_nice, mainly_bad, tit_for_tat, grudger, tit_for_2_tats

strategies = [nice_guy, bad_guy, mainly_nice, mainly_bad, tit_for_tat, grudger, tit_for_2_tats]

print("select the task to be performed :")
print("Task 1: \n Implement a simple IPD between two players implementing two given strategies. \n Study the evolution along the tournament confronting different strategies; \n study the overall outcome in the different configurations.")
print("Task 2: \n Implement a multiple players IPD (MPIPD) where several strategies play against each other in a roud-robin scheme")
print("Task 3: \n Iterate what done in the task_2 (repeated MPIPD, rMPIPD) by increasing the population \n implementing a given strategy depending on the results that strategy achieved in the previous iteration")
# print("Task 4: \n Implement a rMPIPD where strategies are allowed to mutate. \n The goal is to simulate the effect of genetic mutations and the effect of natura selection. \n A parameter (gene) should encode the attidue of an individual to cooperate, such gene can mutate randomly and \n the corresponding phenotype should compete in the MPIPD such that the best-fitted is determined.")
task_num = int(input())

def main():
    if(task_num == 1):
        task_one()
    if(task_num == 2):
        task_two()
    if(task_num == 3):
        task_three()
    
def task_one():
    number_of_players = 2
    players_strategies = [0]*number_of_players
    strategies_length = len(strategies)
    all_section = ''
    i = 0
    while(i < strategies_length):
        players_strategies[0] = strategies[i]
        j = i
        while(j < strategies_length):
            players_strategies[1] = strategies[j]
            modules = players_strategies
            scores, moves = tournament(modules)
            plt.title('Result of the game between '+modules[0].strategy_name+' and '+modules[1].strategy_name)
            plot_graph(scores,modules, colors = ['blue','red'], width = 0.2)
            all_section = create_reports(modules, scores, moves, all_section)
            j = j+1
        i = i+1
    post_to_file(all_section,'1')

def task_two():
    all_section = ''
    modules = strategies
    scores, moves = tournament(modules)
    plt.title('Result of all strategies towards all other strategies')
    plot_graph(scores,modules,colors = ['green'], width = 0.4)
    all_section = create_reports(modules, scores, moves,all_section)
    post_to_file(all_section,'2')

def task_three():
    all_section = ''
    number_of_players = int(input('Enter the number of players \n'))
    if(number_of_players > 1):
        number_of_repeatitions = int(input('Enter number of repeatitions to be performed \n'))
        if(number_of_repeatitions < 1):
            print("Please enter valid number of repeatitions to be performed which should be > 0")
            number_of_repeatitions = int(input('Enter number of repititions to be performed \n'))
            if(number_of_repeatitions < 1):
                print("Invalid enter of repeatitions please start the tournament again \n")
                return ""
        print("select the strategies for all players")
        print("1: nice guy \n 2: bad guy \n 3: mainly nice guy \n 4: mainly bad guy \n 5: tit for tat \n 6: Grudger \n 7: tit for 2 tats \n")
        players_strategies = [0]*number_of_players
        for each_player in range(number_of_players):
            print("select strategy for player "+str(each_player))
            strategy_number = int(input())
            players_strategies[each_player] = strategies[strategy_number-1]
        current_repeatition = 0
        while(current_repeatition < number_of_repeatitions):
            modules = players_strategies
            scores, moves = tournament(modules)
            plt.title('Result of the game ')
            plot_graph(scores,modules, colors = ['orange','cyan'], width = 0.2)
            all_section = create_reports(modules, scores, moves,all_section)
            if current_repeatition < number_of_repeatitions-1:
                print("Enter 0: to change the strategies of the players \n 1: to continue with existing strategies \n ")
                change_strategy = int(input())
                if(change_strategy == 0):
                    print("1: nice guy \n 2: bad guy \n 3: mainly nice guy \n 4: mainly bad guy \n 5: tit for tat \n 6: Grudger \n 7: tit for 2 tats \n")
                    players_strategies = [0]*number_of_players
                    for each_player in range(number_of_players):
                        print("select strategy for player "+str(each_player))
                        strategy_number = int(input())
                        players_strategies[each_player] = strategies[strategy_number-1]
            current_repeatition = current_repeatition + 1
    else:
        print("Tournament is not possible with "+str(number_of_players)+" players")
    post_to_file(all_section,'3')

# def task_four():
#     number_of_players = int(input('Enter the number of players \n'))
#     number_of_repeatitions = int(input('Enter number of repeatitions to be performed \n'))
#     players_strategies = [0]*number_of_players
#     strategies = [mainly_nice, nice_guy, tit_for_tat]
#     for each_player in range(number_of_players):
#         strategy = random.choice(strategies)
#         players_strategies[each_player] = strategy
#     current_repeatition = 0
#     while(current_repeatition < number_of_repeatitions):
#         modules = players_strategies
#         tournament(modules)
#         #task 4 is pending
#         current_repeatition = current_repeatition + 1


def tournament(modules):
    for module in modules:
            reload(module)
            for required_variable in ['strategy_name', 'strategy_description']:
                if not hasattr(module, required_variable):
                    setattr(module, required_variable, 'missing assignment')

    scores, moves = prisoners_dilemma.main_play(modules)
    return scores, moves
    
def create_reports(modules, scores, moves, all_section):
    section0, section1, section2 = prisoners_dilemma.make_reports(modules, scores, moves)
    print(section0+section1+section2)
    prev_all_section = all_section
    all_section = section0 + section1 + section2
    all_section = prev_all_section + all_section
    return all_section

def post_to_file(string, number, filename='', directory=''):
    filename = 'tournament_'+number+'.txt'
    if directory=='':
        directory = os.path.dirname(os.path.abspath(__file__))  
    filename = os.path.join(directory, filename)
    filehandle = open(filename,'w')
    filehandle.write(string)


def plot_graph(scores,modules,colors,width):
    left = list(range(1,len(modules)+1))
    height = [0]*len(modules)
    players_strategies = [0]*len(modules)
    i = 0
    for each_player_score in scores:
        height[i] = sum(each_player_score)/len(modules)
        players_strategies[i] = modules[i].strategy_name
        i = i+1
    plt.bar(left,height, tick_label = players_strategies, width = width, color = colors)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()


if __name__ == '__main__':
    main()