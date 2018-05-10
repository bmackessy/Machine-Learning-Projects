from copy import deepcopy
import GameState
import Game


class InitialGameState:
    def __init__(self, startConfig):
        # Initializes the board with the correct configuration
        # Input:    startConfig

        self.whiteScore = 2
        self.blackScore = 2
        self.totalChips = 4
        self.boardArr = []
        self.validMoves = ['C4', 'E6', 'F5', 'D3']

        # Setup the Board
        for i in range(8):
            row = []
            for j in range(8):
                row.append(' ')
            self.boardArr.append(row)

        self.boardArr[3][3] = 'W'
        self.boardArr[3][4] = 'B'
        self.boardArr[4][3] = 'B'
        self.boardArr[4][4] = 'W'


    def printBoard(self):
        # Prints the current state of the board

        print '    A   B   C   D   E   F   G   H'
        print '  ---------------------------------'
        for i in range(len(self.boardArr)):
            rowString = str(i + 1) + ' | '
            for j in range(len(self.boardArr[i])):
                rowString = rowString + self.boardArr[i][j] + ' | '
            print rowString
            print '  ----------------------------------'
        print ''

    def printScoreBoard(self):
        # Prints the Scoreboard which shows Black's and white's current score
        print  "Score"
        print  "Black " + str(self.blackScore)
        print  "White " + str(self.whiteScore)
        print ''
        print ''

    def getBoardArr(self):
        return deepcopy(self.boardArr)

    def getWhiteScore(self):
        return self.whiteScore

    def getBlackScore(self):
        return self.blackScore

    def getTotalChips(self):
        return self.totalChips

    def getValidMoves(self):
        return self.validMoves

    def flipBoard(self):
        self.boardArr[3][3] = 'B'
        self.boardArr[3][4] = 'W'
        self.boardArr[4][3] = 'W'
        self.boardArr[4][4] = 'B'
        self.validMoves = ['F4', 'E3', 'D6', 'C5']
