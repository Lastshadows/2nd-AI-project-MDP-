# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions, GhostRules
import numpy as np
from pacman_module import util
import random
import csv
import numpy
import scipy.stats as sstat


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        """
            Variables to use in 'updateAndFetBeliefStates' method.
            Initialization occurs in 'get_action' method.
        """
        # Current list of belief states over ghost positions
        self.beliefGhostStates = None
        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None
        # Uniform distribution size parameter 'w'
        # for sensor noise (see instructions)
        self.w = self.args.w
        # Probability for 'leftturn' ghost to take 'EAST' action
        # when 'EAST' is legal (see instructions)
        self.p = self.args.p
        self.time_step = 0
        self.convergences = []

    def _move(self, currPos, delta):
        return tuple([sum(x) for x in zip(currPos, delta)])


    def _isLegal(self, pos):
        return not self.walls[pos[0]][pos[1]]

    def _updatePositionDistribution(self, ghostPos, newBeliefState,
                                   oldGhostProb):

        if not self._isLegal(ghostPos):
            return
        move = {'east': (1, 0), 'south': (0, -1), 'north': (0, 1),
                'west': (-1, 0)}
        east = self._move(ghostPos, move['east'])
        south = self._move(ghostPos, move['south'])
        north = self._move(ghostPos, move['north'])
        west = self._move(ghostPos, move['west'])

        nLegalPos = 0
        for direction in move:
            if self._isLegal(self._move(ghostPos, move[direction])):
                nLegalPos += 1
        p = self.p
        if self._isLegal(east):
            newBeliefState[east] += (p+(1-p)*1/nLegalPos)*oldGhostProb
        if self._isLegal(west):
            newBeliefState[west] += ((1-p)* 1/nLegalPos)*oldGhostProb
        if self._isLegal(north):
            newBeliefState[north] += ((1-p)*1/nLegalPos)*oldGhostProb
        if self._isLegal(south):
            newBeliefState[south] += ((1-p)*1/nLegalPos)*oldGhostProb

        return newBeliefState #.normalize()


    def _normalize(self, pMatrix):

        sum = 0
        for rows in range(pMatrix.shape[0]):
            # sum += rows.sum()
            sum += pMatrix[rows].sum()

        if sum != 0:
            for x in range(pMatrix.shape[0]):
                for y in range(pMatrix.shape[1]):
                    pMatrix[x, y] /= sum
        return pMatrix

    def _computeSensorProb(self):
        W = 2*self.w+1
        return 1/(W**2)

    def _withinSensorRange(self, evidence, pos):
        x, y = pos[0], pos[1]
        validXRange = (x <= evidence[0] + self.w) and (x >= evidence[0]-self.w)
        validYRange = (y <= evidence[1] + self.w) and (y >= evidence[1]-self.w)
        return validXRange and validYRange



    def updateAndGetBeliefStates(self, evidences):
        """
        Bref,
        - on fait une copie du belief state,
        - on boucle sur toutes les positions du belief state,
               - on prend la valeur de la copie, on met à jour les voisins de belief_state avec le transition model sur base de cette valeur.

        - Et puis, pour le terme P(e_t+1|X_t+1), on parcourt tous notre belief state maintenant mis à jours et on multiplie par 1/9 aux elements compris dans le carré W X W centré autour de l'évidence et les autres on met 0.
        - Finalement, on normalize
        """

        beliefStates = self.beliefGhostStates

        # XXX: Your code here -----------------------------------------------

        oldBeliefStates = beliefStates.copy()
        newBeliefState = oldBeliefStates[0].copy()
        newBeliefState[newBeliefState != 0] = 0
        sensorProb = self._computeSensorProb()

        # posDistributions = [beliefs for i, beliefs in enumerate(beliefStates)]
        walls = self.walls
        width = oldBeliefStates[0].shape[0]
        height = oldBeliefStates[0].shape[1]

        for x in range(width):
            for y in range(height):
                p = oldBeliefStates[0][x, y]
                self._updatePositionDistribution((x, y), newBeliefState, p)

        for x in range(width):
            for y in range(height):
                if self._withinSensorRange(evidences[0], (x,y)):
                    beliefStates[0][x, y] *= newBeliefState[x,y] * sensorProb
                else:
                    beliefStates[0][x, y] = 0
        beliefStates[0] = self._normalize(beliefStates[0])

        # XXX: End of your code
        self.beliefGhostStates = beliefStates
        return beliefStates

    def compute_convergence(self):
        # Entropie = convergence = sum(element
        # belief
        # State / log(el
        # belief
        # state)) pour chaque time step
        sum = 0
        beliefState = self.beliefGhostStates
        width = beliefState[0].shape[0]
        height = beliefState[0].shape[1]
        for x in range(width):
            for y in range(height):
                a = beliefState[0][x, y]
                sum += a/numpy.log(a)
        return sum

    def _computeNoisyPositions(self, state):
        """
            Compute a noisy position from true ghosts positions.
            XXX: DO NOT MODIFY THAT FUNCTION !!!
            Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        w = self.args.w
        w2 = 2*w+1
        div = float(w2 * w2)
        new_positions = []
        for p in positions:
            (x, y) = p
            dist = util.Counter()
            for i in range(x - w, x + w + 1):
                for j in range(y - w, y + w + 1):
                    dist[(i, j)] = 1.0 / div
            dist.normalize()
            new_positions.append(util.chooseFromDistribution(dist))
        return new_positions



    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """

        # XXX : You shouldn't care on what is going on below.
        # Variables are specified in constructor.


        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()


        self.time_step += 1

        seq = self.beliefGhostStates[0].reshape(-1)
        entropy = sstat.entropy(seq)
        print(entropy)

        if self.time_step <= 100:
            # self.convergences.append(self.compute_convergence())
            self.convergences.append(entropy)
        if self.time_step == 50:
            with open('convergences.csv', 'w') as csvfile:
                wr = csv.writer(csvfile, dialect='excel')
                results = self.convergences
                wr.writerow(results)
            print("all went good")
            print(self.convergences)


        return self.updateAndGetBeliefStates(
            self._computeNoisyPositions(state))

# --agentfile pacmanagent.py --layout observer --nghosts 1 --ghostagent greedy
# --bsagentfile beliefstateagent.py --p 0.5 --w 1

# --agentfile pacmanagent.py --layout observer
# --bsagentfile beliefstateagent.py --p 0.5 --w 1

# --bsagentfile beliefstateagent --layout observer  --p 0.5 --w 1

    # Proba une des neufs autour
    # O = sur la diagonale , proba evidence(position x, y) sachant
    # F = msg forward
    # T truc que l'on update avec east, ...'