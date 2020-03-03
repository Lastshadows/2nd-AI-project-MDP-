# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions, GhostRules
import numpy as np
from pacman_module import util
import csv
import scipy.stats as sstat
import matplotlib.pyplot as plt

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

        self.time_step = 0
        self.convergences = []

    def _move(self, currPos, delta):
        return tuple([sum(x) for x in zip(currPos, delta)])


    def _isLegal(self, pos):
        return not self.walls[pos[0]][pos[1]]

    def _updatePositionDistribution(self, pos, newBeliefState,
                                    oldGhostProb):

        if not self._isLegal(pos):
            return
        move = {'east': (1, 0), 'south': (0, -1), 'north': (0, 1),
                'west': (-1, 0)}
        east = self._move(pos, move['east'])
        south = self._move(pos, move['south'])
        north = self._move(pos, move['north'])
        west = self._move(pos, move['west'])

        # oldGhostProb = 1

        nLegalPos = 0
        for direction in move:
            if self._isLegal(self._move(pos, move[direction])):
                nLegalPos += 1
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

        return newBeliefState  # .normalize()

    def _normalize(self, pMatrix):

        sum = 0
        for rows in range(pMatrix.shape[0]):
            sum += pMatrix[rows].sum()

        if sum != 0:
            for x in range(pMatrix.shape[0]):
                for y in range(pMatrix.shape[1]):
                    pMatrix[x, y] /= sum
        return pMatrix

    def _computeSensorProb(self):
        W = 2 * self.w + 1
        return 1 / (W ** 2) - 1

    def _withinSensorRange(self, evidence, pos):
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


        #-------------------------------------------------------------------
        #if self.timeStep <= 101:
            #print(self.timeStep)
           # self.timeStep += 1

            #for noGhost in range(len(beliefStates)):
               # seq = beliefStates[noGhost].reshape(-1)
                #self.entropy[noGhost, self.timeStep-1] = sstat.entropy(seq)
                #if self.timeStep == 101:
                   # print("alive")
                    #plt.plot(np.mean(self.entropy, axis=0))
                   # plt.xlabel('Time step')
                   # plt.ylabel('Entropy')
                    #plt.ylim([0, 4])

                    # strings = ["%.5f" % number for number in self.entropy]
                    # csv = open("entropy.csv", "w")
                    # for e in strings:
                    #     csv.write(e + "\n")


                # with open("entropy.csv", 'wb') as resultFile:
                #     wr = csv.writer(resultFile, dialect='excel')
                #     wr.writerow(self.entropy)

        # XXX: Your code here -----------------------------------------------
        oldBeliefStates = beliefStates.copy()
        sensorProb = self._computeSensorProb()

        for noGhost in range(len(oldBeliefStates)):
            newBeliefState = oldBeliefStates[noGhost].copy()
            # Start with only zeroes
            newBeliefState[newBeliefState != 0] = 0
            # posDistributions = [beliefs for i, beliefs in enumerate(beliefStates)]

            width = oldBeliefStates[noGhost].shape[0]
            height = oldBeliefStates[noGhost].shape[1]

            for x in range(width):
                for y in range(height):
                    p_old = oldBeliefStates[noGhost][x, y]  # p(x_t|e_1:t)
                    # For each possible position from (x,y), we update the
                    # probability via the transition model * p_old
                    self._updatePositionDistribution((x, y), newBeliefState,
                                                     p_old)

            for x in range(width):
                for y in range(height):
                    if self._withinSensorRange(evidences[noGhost], (x, y)):
                        beliefStates[noGhost][x, y] = newBeliefState[x,
                                                                     y] * sensorProb
                        # + au lieu d'un *
                    else:
                        beliefStates[noGhost][x, y] = 0

            # beliefStates[noGhost] *= self._computeAlpha(beliefStates[noGhost], evidences[0],
            #                            evidences[1])
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


        self.time_step += 1
        seq = self.beliefGhostStates[0].reshape(-1)
        entropy = sstat.entropy(seq)
        #print(entropy)

        if self.time_step <= 100:
            # self.convergences.append(self.compute_convergence())
            self.convergences.append(entropy)
        if self.time_step == 100:
            with open('convergences.csv', 'w') as csvfile:
                wr = csv.writer(csvfile, dialect='excel')
                results = self.convergences
                wr.writerow(results)
            print("all went good")
            #print(self.convergences)
            plt.plot(self.convergences)
            plt.xlabel('Time step')
            plt.ylabel('Entropy')
            #plt.ylim([0, 4])
            #plt.show()

            print("graph done")
            print(self.convergences)




        return self.updateAndGetBeliefStates(
            self._computeNoisyPositions(state))
