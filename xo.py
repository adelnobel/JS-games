import random
import copy
import json
import time
import sys

class AbstractGame:
    def getNextStateFromStateAndAction(self, state, action):
        return NotImplemented
    def getInitialState(self, state, action):
        return NotImplemented
    def isTerminalState(self, state):
        return NotImplemented
    def getRewardForStateAndAction(self, state, action):
        return NotImplemented
    def getValidActionsForState(self, state):
        return NotImplemented


class XOGame(AbstractGame):
    def getNextStateFromStateAndAction(self, state, action):
        next_state = copy.deepcopy(state)
        next_state[1] = 'O' if state[1] == 'X' else 'X'
        next_state[0][action[0]][action[1]] = state[1]
        return next_state

    def getInitialState(self):
        return [[['.', '.', '.'],['.', '.','.'],['.', '.', '.']], 'X']

    def isTerminalState(self, state):
        board = state[0]
        winner = self.getBoardWinner(board)
        if winner is not None:
            return True
        return self.__isAllFilled(board)

    def getRewardForStateAndAction(self, state, action):
        assert not self.isTerminalState(state)
        next_state = self.getNextStateFromStateAndAction(state, action)
        if self.isTerminalState(next_state):
            board = next_state[0]
            return 100 if self.getBoardWinner(board) is not None else 0
        return 0

    def getValidActionsForState(self, state):
        board = state[0]
        actions = []
        for r in range(3):
            for c in range(3):
                if board[r][c] == '.':
                    actions.append((r, c))
        return actions

    def printBoardFromState(self, state):
        board = state[0]
        for i in range(3):
            print board[i]
        print '---------------'

    def __isAllFilled(self, board):
        return all([all([col != '.' for col in row]) for row in board])

    def getBoardWinner(self, board):
        for i in range(3):
            col_winner = self.__getColWinner(board, i)
            if col_winner is not None:
                return col_winner
            row_winner = self.__getRowWinner(board, i)
            if row_winner is not None:
                return row_winner
        return self.__getDiagonalsWinner(board)

    def __getRowWinner(self, board, row_idx):
        if board[row_idx][0] == board[row_idx][1] and board[row_idx][0] == board[row_idx][2]:
            return board[row_idx][0] if board[row_idx][0] != '.' else None
        return None

    def __getColWinner(self, board, col_idx):
        if board[0][col_idx] == board[1][col_idx] and board[0][col_idx] == board[2][col_idx]:
            return board[0][col_idx] if board[0][col_idx] != '.' else None
        return None

    def __getDiagonalsWinner(self, board):
        if board[0][0] == board[1][1] and board[0][0] == board[2][2]:
            return board[0][0] if board[0][0] != '.' else None
        if board[0][2] == board[1][1] and board[0][2] == board[2][0]:
            return board[0][2] if board[0][2] != '.' else None
        return None


class Agent:
    def makeAction(self, state, game):
        return NotImplemented

    def observe(self, state, score, game):
        if game.isTerminalState(state):
            return None
        return self.makeAction(state, game)

class RandomAgent(Agent):
    def makeAction(self, state, game):
        actions = game.getValidActionsForState(state)
        return random.choice(actions)

class HumanAgent(Agent):
    def makeAction(self, state, game):
        inp = input()
        return inp

class MinMaxAgent(Agent):
    def __init__(self):
        self.memo = {}

    def makeAction(self, state, game):
        return self.__getBestActionAndValue(state, game)[0]

    def __minMax(self, state, game):
        assert not game.isTerminalState(state)
        return self.__getBestActionAndValue(state, game)

    def __getBestActionAndValue(self, state, game):
        serialized_state = json.dumps(state)
        if serialized_state in self.memo:
            return self.memo[serialized_state]
        actions = game.getValidActionsForState(state)
        actions_values = []
        for action in actions:
            action_value = 0
            next_state = game.getNextStateFromStateAndAction(state, action)
            if game.isTerminalState(next_state):
                action_value = game.getRewardForStateAndAction(state, action)
            else:
                action_value = -self.__minMax(next_state, game)[1]
            actions_values.append(action_value)
        best_value = max(actions_values)
        self.memo[serialized_state] = (actions[actions_values.index(best_value)], best_value)
        return (actions[actions_values.index(best_value)], best_value)



class QLearnerModelAgent(Agent):

    def __init__(self):
        self.q_values = {}
        self.v_values = {}
        self.policy = {}
        self.episode = []
        self.discount = 1
        self.learning_rate = 0.3
        self.greedy_eps = 0.4

    def makeAction(self, state, game):
        actions = game.getValidActionsForState(state)
        random_val = random.uniform(0, 1)
        if (random_val <= self.greedy_eps):
            action = random.choice(actions)
        else:
            action = self.__getPolicyAction(state, actions)
        self.episode.append(action)
        return action

    def __getPolicyAction(self, state, actions):
        serialized_state = json.dumps(state)
        if serialized_state not in self.policy:
            self.policy[serialized_state] = random.choice(actions)
        return self.policy[serialized_state]


    def observe(self, state, score, game):
        self.episode.append((state, score))
        if game.isTerminalState(state):
            self.__learnFromEpisode()
            return None
        return self.makeAction(state, game)

    def __learnFromEpisode(self):
        episode = list(reversed(self.episode))
        self.episode = []
        self.v_values[json.dumps(episode[0][0])] = 0 #terminal state

        for i in range(1, len(episode), 2):
            action = episode[i]
            state = episode[i + 1][0]
            score = episode[i + 1][1]
            next_state = episode[i - 1][0]
            next_score = episode[i - 1][1]

            assert not game.isTerminalState(state)
            reward = next_score - score

            if json.dumps(state) not in self.q_values:
                self.q_values[json.dumps(state)] = {}

            if json.dumps(action) not in self.q_values[json.dumps(state)]:
                self.q_values[json.dumps(state)][json.dumps(action)] = 0

            old_q_value = self.q_values[json.dumps(state)][json.dumps(action)]

            self.q_values[json.dumps(state)][json.dumps(action)] = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + self.discount * self.v_values[json.dumps(next_state)])

            all_actions = game.getValidActionsForState(state)
            actions_q_values = [self.q_values[json.dumps(state)].get(json.dumps(action), 0) for action in all_actions]
            self.v_values[json.dumps(state)] = max(actions_q_values)
            self.policy[json.dumps(state)] = all_actions[actions_q_values.index(max(actions_q_values))]

        self.greedy_eps -= 0.005
        self.learning_rate -= 0.00001
        if self.learning_rate < 0:
            self.learning_rate = 0

players = [QLearnerModelAgent(), RandomAgent()]
game = XOGame()
'''print players[1].observe([[['X', 'O', 'X'],['O', 'O', '.'],['.', 'X', '.']], 'O'], 0, game)

sys.exit(0)'''

totalx = 0
totalo = 0
total_games = 10000
for x in range(2):
    for i in range(total_games):
        game = XOGame()
        current_state = game.getInitialState()
        player_in_turn = 0

        while not game.isTerminalState(current_state):
            player_action = players[player_in_turn].observe(current_state, 0, game)
            current_state = game.getNextStateFromStateAndAction(current_state, player_action)
            if x == 1:
                game.printBoardFromState(current_state)
                if player_in_turn == 1:
                    print players[0].v_values.get(json.dumps(current_state), 0)
            player_in_turn += 1
            player_in_turn %= 2

        winner = game.getBoardWinner(current_state[0])
        score = 0
        if winner == 'X':
            score = 10000
            totalx += 1
        elif winner == 'O':
            score = -100
            totalo += 1
        players[0].observe(current_state, score , game)

        if x == 1:
            print winner, ' wins'


    print 'X wins', totalx
    print 'O wins', totalo

    print 1.0 * totalo / total_games
    players[1] = HumanAgent()
    players[0].learning_rate = 0.7
