# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions, GhostRules
import numpy as np
from pacman_module import util
import csv
import scipy.stats as sstat
import matplotlib.pyplot as plt
import pandas as pd


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

        self.timeStep = 0
        self.entropy = np.zeros((5, 101))

    def _move(self, currPos, delta):
        '''
        Perform a move from a position and return the corresponding position
        :param currPos: (x,y)
        :param delta: (dx, dy)
        :return: newPos
        '''
        return tuple([sum(x) for x in zip(currPos, delta)])

    def _isLegal(self, pos):
        '''
        Check wether a position is legal with regards the walls
        :param pos: (x,y)
        :return: bool
        '''
        return not self.walls[pos[0]][pos[1]]

    def _updatePositionDistribution(self, pos, newBeliefState,
                                    oldGhostProb):
        '''
        Update the distribution of probabilities with the transition model
        :param pos: (x,y)
        :param newBeliefState: the belief state that we update
        :param oldGhostProb: P(x_t|e_1:t)
        :return: update new_belief_state
        '''

        if not self._isLegal(pos):
            return
        move = {'east': (1, 0), 'south': (0, -1), 'north': (0, 1),
                'west': (-1, 0)}
        east = self._move(pos, move['east'])
        south = self._move(pos, move['south'])
        north = self._move(pos, move['north'])
        west = self._move(pos, move['west'])

        # Compute the number of legal surrounding position
        nLegalPos = 0
        for direction in move:
            if self._isLegal(self._move(pos, move[direction])):
                nLegalPos += 1

        # Transition model
        p = self.p
        if self._isLegal(east):
            newBeliefState[east] += (p + (
                        1 - p) * 1 / nLegalPos) * oldGhostProb
        if self._isLegal(west):
            newBeliefState[west] += ((1 - p) * 1 / nLegalPos) * oldGhostProb
        if self._isLegal(north):
            newBeliefState[north] += ((1 - p) * 1 / nLegalPos) * oldGhostProb
        if self._isLegal(south):
            newBeliefState[south] += ((1 - p) * 1 / nLegalPos) * oldGhostProb

        return newBeliefState

    def _normalize(self, pMatrix):
        '''
        Normalization of a belief state. There is nos special need to use
        the probabilistic expression given in the course though.
        :param pMatrix: belief state
        :return: updated belief state
        '''
        sum = 0
        for rows in range(pMatrix.shape[0]):
            sum += pMatrix[rows].sum()

        if sum != 0:
            for x in range(pMatrix.shape[0]):
                for y in range(pMatrix.shape[1]):
                    pMatrix[x, y] /= sum
        return pMatrix

    def _computeSensorProb(self):
        '''
        Get the uniform distribution (proba.) of a tile covered by the
        sensor
        :return: the probability of being in one tile
        '''
        W = 2 * self.w + 1
        return 1 / (W ** 2)

    def _withinSensorRange(self, evidence, pos):
        '''
        Check wether a position is within the sensor range
        :param evidence: position detected by the sensor
        :param pos: position considered
        :return: bool
        '''
        x, y = pos[0], pos[1]
        validXRange = (x <= evidence[0] + self.w) and (
                    x >= evidence[0] - self.w)
        validYRange = (y <= evidence[1] + self.w) and (
                    y >= evidence[1] - self.w)
        return validXRange and validYRange

    def updateAndGetBeliefStates(self, evidences):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of (noised) ghost positions at state x_{t}
          where 't' is the current time step

        Return:
        -------
        - A list of Z belief states at state x_{t} about ghost positions
          as N*M numpy matrices of probabilities
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """

        beliefStates = self.beliefGhostStates

        #  -- PLOT --
        if self.timeStep < 101:
            self.timeStep += 1
            for noGhost in range(len(beliefStates)):
                seq = beliefStates[noGhost].reshape(-1)
                self.entropy[noGhost, self.timeStep-1] = sstat.entropy(seq)
        if self.timeStep == 101:
            self.timeStep += 1
            res = np.mean(self.entropy, axis=0)
            # plt.plot(res)
            # plt.xlabel('Time step')
            # plt.ylabel('Entropy')
            # plt.show()
            df = pd.DataFrame(res)
            df.to_csv("entropy_p0_w5_1.csv")
            exit()

        # -- Prediction and update --
        sensorProb = self._computeSensorProb()

        for noGhost in range(len(beliefStates)):
            newBeliefState = beliefStates[noGhost].copy()

            # Start with only zeroes
            newBeliefState[newBeliefState != 0] = 0
            width = beliefStates[noGhost].shape[0]
            height = beliefStates[noGhost].shape[1]

            for x in range(width):
                for y in range(height):
                    p_old = beliefStates[noGhost][x, y]  # p(x_t|e_1:t)
                    # For each possible position from (x,y), we update the
                    # probability via the transition model * p_old
                    self._updatePositionDistribution((x, y), newBeliefState,
                                                     p_old)

            for x in range(width):
                for y in range(height):
                    if self._withinSensorRange(evidences[noGhost], (x, y)):
                        beliefStates[noGhost][x, y] = newBeliefState[x,
                                                                     y] * \
                                                      sensorProb
                    else:
                        beliefStates[noGhost][x, y] = 0

            beliefStates[noGhost] = self._normalize(beliefStates[noGhost])

        # XXX: End of your code

        self.beliefGhostStates = beliefStates
        return beliefStates

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
        return self.updateAndGetBeliefStates(
            self._computeNoisyPositions(state))
