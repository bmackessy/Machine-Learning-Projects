from copy import deepcopy
#import colorama
#from colorama import Fore, Back, Style

class GameState:

    def __init__(self, prevState, move, player, nextPlayer):
        self.playerTurn = player  # W or B
        self.move = move
        self.whiteScore = prevState.getWhiteScore()
        self.blackScore = prevState.getBlackScore()
        self.boardArr = prevState.getBoardArr()
        self.totalChips = prevState.getTotalChips()
        self.validMoves = []
        self.nextPlayer = nextPlayer
        self.newTiles = []

        if move == 'null':
            self.setValidMoves()
        else:
            # Parse Move
            xIdx, yIdx = self.parseMove(self.move)

            # Place Token and Add 1 to score
            self.boardArr[yIdx][xIdx] = self.playerTurn
            if player == "W":
                self.whiteScore += 1
            else:
                self.blackScore += 1

            #Add new token to newTiles list
            self.newTiles.append(self.parseMove(self.move))

            # Reconfigure Board and update score with Parsed Move
            self.boardArr[yIdx][xIdx] = self.playerTurn
            self.config_N(xIdx, yIdx)
            self.config_NE(xIdx, yIdx)
            self.config_E(xIdx, yIdx)
            self.config_SE(xIdx, yIdx)
            self.config_S(xIdx, yIdx)
            self.config_SW(xIdx, yIdx)
            self.config_W(xIdx, yIdx)
            self.config_NW(xIdx, yIdx)

            # Recalculate Total Chips
            self.totalChips += 1





    def config_N(self, xIdx, yIdx):
        player = self.playerTurn
        foundEnd = False
        foundPlayer = False
        yIdx -= 1
        potentialFlips = []
        while foundEnd == False and foundPlayer == False:
            # If found edge of board
            if yIdx == -1:
                foundEnd = True

            # Else if found blank space
            elif self.boardArr[yIdx][xIdx] == ' ':
                foundEnd = True

            # Else if found same player
            elif self.boardArr[yIdx][xIdx] == player:
                foundPlayer = True

            # Else found opposite player
            else:
                potentialFlips.append([xIdx, yIdx])
                yIdx -= 1

        if foundPlayer:
            for flip in potentialFlips:
                self.newTiles.append(flip)

    def config_NE(self, xIdx, yIdx):
        player = self.playerTurn
        foundEnd = False
        foundPlayer = False
        yIdx -= 1
        xIdx += 1
        potentialFlips = []
        while foundEnd == False and foundPlayer == False:
            # If found edge of board
            if yIdx == -1 or xIdx == 8:
                foundEnd = True

            # Else if found blank space
            elif self.boardArr[yIdx][xIdx] == ' ':
                foundEnd = True

            # Else if found same player
            elif self.boardArr[yIdx][xIdx] == player:
                foundPlayer = True

            # Else found opposite player
            else:
                potentialFlips.append([xIdx, yIdx])
                yIdx -= 1
                xIdx += 1

        if foundPlayer:
            for flip in potentialFlips:
                self.newTiles.append(flip)

    def config_E(self, xIdx, yIdx):
        player = self.playerTurn
        foundEnd = False
        foundPlayer = False
        xIdx += 1
        potentialFlips = []
        while foundEnd == False and foundPlayer == False:
            # If found edge of board
            if xIdx == 8:
                foundEnd = True

            # Else if found blank space
            elif self.boardArr[yIdx][xIdx] == ' ':
                foundEnd = True

            # Else if found same player
            elif self.boardArr[yIdx][xIdx] == player:
                foundPlayer = True

            # Else found opposite player
            else:
                potentialFlips.append([xIdx, yIdx])
                xIdx += 1

        if foundPlayer:
            for flip in potentialFlips:
                self.newTiles.append(flip)

    def config_SE(self, xIdx, yIdx):
        player = self.playerTurn
        foundEnd = False
        foundPlayer = False
        yIdx += 1
        xIdx += 1
        potentialFlips = []
        while foundEnd == False and foundPlayer == False:
            # If found edge of board
            if yIdx == 8 or xIdx == 8:
                foundEnd = True

            # Else if found blank space
            elif self.boardArr[yIdx][xIdx] == ' ':
                foundEnd = True

            # Else if found same player
            elif self.boardArr[yIdx][xIdx] == player:
                foundPlayer = True

            # Else found opposite player
            else:
                potentialFlips.append([xIdx, yIdx])
                yIdx += 1
                xIdx += 1

        if foundPlayer:
            for flip in potentialFlips:
                self.newTiles.append(flip)

    def config_S(self, xIdx, yIdx):
        player = self.playerTurn
        foundEnd = False
        foundPlayer = False
        yIdx += 1
        potentialFlips = []
        while foundEnd == False and foundPlayer == False:
            # If found edge of board
            if yIdx == 8 or xIdx == -1:
                foundEnd = True

            # Else if found blank space
            elif self.boardArr[yIdx][xIdx] == ' ':
                foundEnd = True

            # Else if found same player
            elif self.boardArr[yIdx][xIdx] == player:
                foundPlayer = True

            # Else found opposite player
            else:
                potentialFlips.append([xIdx, yIdx])
                yIdx += 1

        if foundPlayer:
            for flip in potentialFlips:
                self.newTiles.append(flip)

    def config_SW(self, xIdx, yIdx):
        player = self.playerTurn
        foundEnd = False
        foundPlayer = False
        yIdx += 1
        xIdx -= 1
        potentialFlips = []
        while foundEnd == False and foundPlayer == False:
            # If found edge of board
            if yIdx == 8 or xIdx == -1:
                foundEnd = True

            # Else if found blank space
            elif self.boardArr[yIdx][xIdx] == ' ':
                foundEnd = True

            # Else if found same player
            elif self.boardArr[yIdx][xIdx] == player:
                foundPlayer = True

            # Else found opposite player
            else:
                potentialFlips.append([xIdx, yIdx])
                yIdx += 1
                xIdx -= 1

        if foundPlayer:
            for flip in potentialFlips:
                self.newTiles.append(flip)

    def config_W(self, xIdx, yIdx):
        player = self.playerTurn
        foundEnd = False
        foundPlayer = False
        xIdx -= 1
        potentialFlips = []
        while foundEnd == False and foundPlayer == False:
            # If found edge of board
            if yIdx == -1 or xIdx == -1:
                foundEnd = True

            # Else if found blank space
            elif self.boardArr[yIdx][xIdx] == ' ':
                foundEnd = True

            # Else if found same player
            elif self.boardArr[yIdx][xIdx] == player:
                foundPlayer = True

            # Else found opposite player
            else:
                potentialFlips.append([xIdx, yIdx])
                xIdx -= 1

        if foundPlayer:
            for flip in potentialFlips:
                self.newTiles.append(flip)

    def config_NW(self, xIdx, yIdx):
        player = self.playerTurn
        foundEnd = False
        foundPlayer = False
        yIdx -= 1
        xIdx -= 1
        potentialFlips = []
        while foundEnd == False and foundPlayer == False:
            # If found edge of board
            if yIdx == -1 or xIdx == -1:
                foundEnd = True

            # Else if found blank space
            elif self.boardArr[yIdx][xIdx] == ' ':
                foundEnd = True

            # Else if found same player
            elif self.boardArr[yIdx][xIdx] == player:
                foundPlayer = True

            # Else found opposite player
            else:
                potentialFlips.append([xIdx, yIdx])
                yIdx -= 1
                xIdx -= 1

        if foundPlayer:
            for flip in potentialFlips:
                self.newTiles.append(flip)

    def parseMove(self, position):

        position = list(position)

        if position[0] == 'A':
            position[0] = 0
        elif position[0] == 'B':
            position[0] = 1
        elif position[0] == 'C':
            position[0] = 2
        elif position[0] == 'D':
            position[0] = 3
        elif position[0] == 'E':
            position[0] = 4
        elif position[0] == 'F':
            position[0] = 5
        elif position[0] == 'G':
            position[0] = 6
        elif position[0] == 'H':
            position[0] = 7

        if position[1] == '1':
            position[1] = 0
        elif position[1] == '2':
            position[1] = 1
        elif position[1] == '3':
            position[1] = 2
        elif position[1] == '4':
            position[1] = 3
        elif position[1] == '5':
            position[1] = 4
        elif position[1] == '6':
            position[1] = 5
        elif position[1] == '7':
            position[1] = 6
        elif position[1] == '8':
            position[1] = 7

        return position[0], position[1]

    def unParseMove(self, position):
        move = ["A", "A"]
        if position[0] == 0:
            move[0] = 'A'
        elif position[0] == 1:
            move[0] = 'B'
        elif position[0] == 2:
            move[0] = 'C'
        elif position[0] == 3:
            move[0] = 'D'
        elif position[0] == 4:
            move[0] = 'E'
        elif position[0] == 5:
            move[0] = 'F'
        elif position[0] == 6:
            move[0] = 'G'
        elif position[0] == 7:
            move[0] = 'H'

        if position[1] == 0:
            move[1] = '1'
        elif position[1] == 1:
            move[1] = '2'
        elif position[1] == 2:
            move[1] = '3'
        elif position[1] == 3:
            move[1] = '4'
        elif position[1] == 4:
            move[1] = '5'
        elif position[1] == 5:
            move[1] = '6'
        elif position[1] == 6:
            move[1] = '7'
        elif position[1] == 7:
            move[1] = '8'

        return ''.join(move)

    def printBoard(self):
        # Prints the current state of the board

        print '    A   B   C   D   E   F   G   H'
        print '  ---------------------------------'
        for i in range(len(self.boardArr)):
            rowString = str(i + 1) + ' | '
            for j in range(len(self.boardArr[i])):
                if ([j,i] in (self.newTiles) or (j,i) in (self.newTiles)) and ([j+1,i] in (self.newTiles) or (j+1,i) in (self.newTiles)):
                    rowString = rowString + '*' + self.boardArr[i][j] + '*' + '|'
                elif [j,i] in (self.newTiles) or (j,i) in (self.newTiles):
                    rowString = rowString + '*' + self.boardArr[i][j] + '*' + '| '
                elif [j+1,i] in (self.newTiles) or (j+1,i) in (self.newTiles):
                    rowString = rowString + self.boardArr[i][j] + ' |'
                else:
                    rowString = rowString + self.boardArr[i][j] + ' | '
            print rowString
            print '  ---------------------------------'
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

    def setValidMoves(self):

        for i in range(len(self.boardArr)):
            for j in range(len(self.boardArr)):

                try:
                    # if space to the North is an opposing chip
                    if self.boardArr[i - 1][j] == self.playerTurn and self.boardArr[i][j] == ' ':
                        for k in range(i - 1, -1, -1):
                            # look to see if there is friendly chip in direction
                            if self.boardArr[k][j] == self.nextPlayer:
                                self.validMoves.append(self.unParseMove([j, i]))
                                break
                            # a valid move cannot have empty space between pieces
                            if self.boardArr[k][j] == ' ':
                                break
                except:
                    IndexError

                try:
                    # if space to the South is an opposing chip
                    if self.boardArr[i + 1][j] == self.playerTurn and self.boardArr[i][j] == ' ':
                        for k in range(i + 1, len(self.boardArr[i])):
                            # look to see if there is friendly chip in direction
                            if self.boardArr[k][j] == self.nextPlayer:
                                self.validMoves.append(self.unParseMove([j, i]))
                                break
                            if self.boardArr[k][j] == ' ':
                                break

                except:
                    IndexError

                try:
                    # if space to the East is an opposing chip
                    if self.boardArr[i][j + 1] == self.playerTurn and self.boardArr[i][j] == ' ':
                        for k in range(j + 1, len(self.boardArr[j])):
                            # look to see if there is friendly chip in direction
                            if self.boardArr[i][k] == self.nextPlayer:
                                self.validMoves.append(self.unParseMove([j, i]))
                                break
                            if self.boardArr[i][k] == ' ':
                                break

                except:
                    IndexError

                try:
                    # if space to the West is an opposing chip
                    if self.boardArr[i][j - 1] == self.playerTurn and self.boardArr[i][j] == ' ':
                        for k in range(j - 1, -1, -1):
                            # look to see if there is friendly chip in direction
                            if self.boardArr[i][k] == self.nextPlayer:
                                self.validMoves.append(self.unParseMove([j, i]))
                                break
                            if self.boardArr[i][k] == ' ':
                                break
                except:
                    IndexError

                try:
                    # if space to the NW is an opposing chip
                    if self.boardArr[i - 1][j - 1] == self.playerTurn and self.boardArr[i][j] == ' ':
                        temp = i - 1
                        for k in range(j - 1, -1, -1):
                            # look to see if there is friendly chip in direction
                            if self.boardArr[temp][k] == self.nextPlayer:
                                self.validMoves.append(self.unParseMove([j, i]))
                                break
                            if self.boardArr[temp][k] == ' ':
                                break
                            # if on the edge
                            elif temp == 0 or k == 0:
                                break

                            else:
                                temp -= 1

                except:
                    IndexError

                try:
                    # if space to the NE is an opposing chip
                    if self.boardArr[i - 1][j + 1] == self.playerTurn and self.boardArr[i][j] == ' ':
                        temp = i - 1
                        for k in range(j + 1, len(self.boardArr)):
                            # look to see if there is friendly chip in direction
                            if self.boardArr[temp][k] == self.nextPlayer:
                                self.validMoves.append(self.unParseMove([j, i]))
                                break
                            if self.boardArr[temp][k] == ' ':
                                break
                            # if on the edge
                            elif temp == 0:
                                break
                            else:
                                temp -= 1
                except:
                    IndexError

                try:
                    # if space to the SE is an opposing chip
                    if self.boardArr[i + 1][j + 1] == self.playerTurn and self.boardArr[i][j] == ' ':
                        temp = i + 1
                        for k in range(j + 1, len(self.boardArr)):
                            # look to see if there is friendly chip in direction
                            if self.boardArr[temp][k] == self.nextPlayer:
                                self.validMoves.append(self.unParseMove([j, i]))
                                break
                            if self.boardArr[temp][k] == ' ':
                                break
                            # if on the edge
                            elif temp == len(self.boardArr) - 1:
                                break
                            else:
                                temp += 1

                except:
                    IndexError

                try:
                    # if space to the SW is an opposing chip
                    if self.boardArr[i + 1][j - 1] == self.playerTurn and self.boardArr[i][j] == ' ':
                        temp = i + 1
                        for k in range(j - 1, -1, -1):
                            # look to see if there is friendly chip in direction
                            if self.boardArr[temp][k] == self.nextPlayer:
                                self.validMoves.append(self.unParseMove([j, i]))
                                break
                            if self.boardArr[temp][k] == ' ':
                                break
                                # if on the edge
                            elif temp == len(self.boardArr) - 1 or k == 0:
                                break
                            else:
                                temp += 1

                except:
                    IndexError


    def printNewTiles(self):

        print "New Tiles:"
        print self.newTiles
        print ''


    def flipTiles(self):
        for flip in self.newTiles:
            if self.playerTurn == "W":
                self.whiteScore += 1
                self.blackScore -= 1
            else:
                self.blackScore += 1
                self.whiteScore -= 1
            self.boardArr[flip[1]][flip[0]] = self.playerTurn

        if self.playerTurn == "W":
            self.whiteScore -= 1
            self.blackScore += 1
        else:
            self.blackScore -= 1
            self.whiteScore += 1

        self.newTiles = []

        # Set Valid Moves
        self.setValidMoves()

    def calculateScore(self):

        self.blackScore = 0
        self.whiteScore = 0

        for i in range(len(self.boardArr)):
            for j in range(len(self.boardArr)):

                if self.boardArr[i][j] == 'W':
                    self.whiteScore += 1
                if self.boardArr[i][j] == 'B':
                    self.blackScore += 1






