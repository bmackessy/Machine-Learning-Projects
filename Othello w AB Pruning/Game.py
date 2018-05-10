import InitialGameState
import GameState
import sys
import random
from threading import Thread
from time import sleep




class Game:
    def __init__(self):
        self.gamestates = []
        self.gamestates.append(InitialGameState.InitialGameState(0))
        self.currentState = 0
        self.currentTurn = "B"
        self.playerColor = "B"
        self.aiColor = "W"
        self.timerStop = False
        self.outOfTime = False
        self.possibleEndGame = False
        self.possibleEndGameAB = False
        self.forceAMove = False
        self.aiMadeMove = False


        self.gamestates[self.currentState].printBoard()
        self.gamestates[self.currentState].printScoreBoard()

        flipBoard = raw_input('Would you like to flip the configuration of the board? (Y/N)')
        if flipBoard == 'Y':
            self.gamestates[self.currentState].flipBoard()
            print 'Here is the new board configuration'
            self.gamestates[self.currentState].printBoard()
            self.gamestates[self.currentState].printScoreBoard()

        config = raw_input('Would you like to be B or W?')
        print ''

        if config == 'W':
            self.playerColor = "W"
            self.aiColor = "B"



    def getPlayerColor(self):
        return self.playerColor

    def getCurrentState(self):
        return self.gamestates[self.currentState]

    def makeMove(self):
        if self.playerColor == self.currentTurn:
            # Players Turn
            self.playerMove()
        else:
            # AIs Turn
            self.AIMove()

    def playerMove(self):
        print "Your Turn"
        validMoves = self.gamestates[self.currentState].getValidMoves()
        print 'Valid Moves = ' + str(validMoves)
        if validMoves == []:

            if self.possibleEndGame == True:
                self.endGame()
            else:
                print "You have no valid moves!"
                raw_input("Press Enter to confirm and pass your turn")
                self.possibleEndGame = True
                self.gamestates.append(
                    GameState.GameState(self.gamestates[self.currentState], 'null', self.playerColor, self.aiColor))
                self.currentState += 1
                self.currentTurn = self.aiColor
        else:
            self.possibleEndGame = False
            move = raw_input("Make a move: ")
            print "The move you are about to make is: " + move
            raw_input("Press Enter to confirm this move")


            if move == "quit":
                sys.exit()
            elif move == "prev":
                prevBoardNumber = int(raw_input("How many turns ago would you like to see the board for (an integer): "))
                self.gamestates[self.currentState - prevBoardNumber].printBoard()
                self.gamestates[self.currentState - prevBoardNumber].printScoreBoard()
                self.currentTurn = self.playerColor
            else:

                if move in validMoves:
                    self.gamestates.append(GameState.GameState(self.gamestates[self.currentState], move, self.playerColor, self.aiColor))
                    self.currentState += 1
                    self.gamestates[self.currentState].printBoard()
                    raw_input("Press Enter to confirm new board layout")
                    self.gamestates[self.currentState].flipTiles()
                    self.gamestates[self.currentState].calculateScore()
                    self.gamestates[self.currentState].printBoard()
                    self.gamestates[self.currentState].printScoreBoard()
                    self.currentTurn = self.aiColor

                else:
                    print 'Not a Valid Move.. Try Again'

    def AIMove(self):
        validMoves = self.gamestates[self.currentState].getValidMoves()
        print 'Valid Moves = ' + str(validMoves)
        if validMoves == []:

            if self.possibleEndGame == True:
                self.endGame()
            else:
                print "AI has no valid moves"
                raw_input("Press Enter to confirm and pass AI's turn")
                self.gamestates.append(
                    GameState.GameState(self.gamestates[self.currentState], 'null', self.aiColor, self.playerColor))
                self.currentState += 1
                self.currentTurn = self.playerColor
        else:
            self.possibleEndGame = False
            if self.forceAMove == False:
                aiDecision = Thread(target=self.AIDecision)
                aiDecision.start()
            timer = Thread(target=self.timer)
            timer.start()  # Calls first function
            while self.aiMadeMove == False:
                sleep(1)

            print 'AI made a move'
            print self.forceAMove
            self.aiMadeMove = False

            self.gamestates[self.currentState].printBoard()
            raw_input("Press Enter to confirm current board layout")
            self.gamestates[self.currentState].flipTiles()
            self.gamestates[self.currentState].calculateScore()
            self.gamestates[self.currentState].printBoard()
            self.gamestates[self.currentState].printScoreBoard()
            self.currentTurn = self.playerColor

    def AIDecision(self):
        print ("AI is thinking....")

        val, move = self.alphaBetaAI(self.gamestates[self.currentState],'null','null',5)
        if self.forceAMove == False:
            self.timerStop = True
            print "The AI's move is: " + move
            raw_input("Press Enter to confirm this move and view new board layout")
            self.gamestates.append(GameState.GameState(self.gamestates[self.currentState], move, self.aiColor, self.playerColor))
            self.currentState += 1
            self.aiMadeMove = True
        else:
            self.forceAMove = False

    def timer(self):
        for i in range(10):
            if self.forceAMove == True:
                self.aiforceMove()
                break
            if i == 9:
                self.aiforceMove()
                break
            sleep(1)
            if self.timerStop == True:
                break

        if self.timerStop == False:
            print "AI out of time!!!"
            self.outOfTime = True

        self.timerStop = False

    def aiforceMove(self):
        print 'forcing move'
        self.forceAMove = True
        self.timerStop = True
        validMoves = self.gamestates[self.currentState].getValidMoves()
        move = validMoves[0]
        print "AI's move is: " + move
        raw_input("Press Enter to confirm this move and view new board layout")
        self.gamestates.append(
            GameState.GameState(self.gamestates[self.currentState], move, self.aiColor, self.playerColor))
        self.currentState += 1
        self.aiMadeMove = True





    def alphaBetaAI(self,gamestate,alpha,beta,depth):
        if self.forceAMove == True:

            sys.exit()

        if self.outOfTime == True:
            quit()
        if gamestate.getTotalChips() > 4:
            gamestate.flipTiles()
            gamestate.calculateScore()
        validMoves = gamestate.getValidMoves()

        if depth == 0:
            return self.mobilityHeuristic(gamestate), "Null"
            #return self.heuristic(gamestate), "Null"

        if validMoves == []:
            if self.possibleEndGameAB == True:
                return self.heuristic(gamestate), "Null"
            else:
                cmpVal, cmpMove = self.alphaBetaOpponent(GameState.GameState(gamestate, 'null', self.aiColor, self.playerColor), alpha,
                                       beta, depth - 1)
                return cmpVal, cmpMove

        self.possibleEndGameAB = False
        bestVal = float("-inf")
        bestMove = validMoves[0]
        for move in validMoves:
            cmpVal, cmpMove = self.alphaBetaOpponent(GameState.GameState(gamestate, move, self.aiColor, self.playerColor), alpha, beta, depth - 1)
            if cmpVal > bestVal:
                bestVal = cmpVal
                bestMove = move
            if beta != 'null':
                if cmpVal >= beta:
                    return bestVal, bestMove
            if alpha == 'null' or cmpVal > alpha:
                alpha = cmpVal
        return bestVal, bestMove

    def alphaBetaOpponent(self,gamestate,alpha,beta,depth):
        if self.forceAMove == True:
            sys.exit()

        if self.outOfTime == True:
            quit()
        gamestate.flipTiles()
        gamestate.calculateScore()
        validMoves = gamestate.getValidMoves()

        if depth == 0:
            return self.heuristic(gamestate), "Null"


        #Checks for a pass or an end of game
        if validMoves == []:
            if self.possibleEndGameAB == True:
                return self.heuristic(gamestate), "Null"
            else:
                cmpVal, cmpMove = self.alphaBetaOpponent(GameState.GameState(gamestate, 'null', self.playerColor, self.aiColor), alpha,
                                       beta, depth - 1)
                return cmpVal, cmpMove

        self.possibleEndGameAB = False
        bestVal = float("inf")
        bestMove = validMoves[0]
        for move in validMoves:
            cmpVal, cmpMove = self.alphaBetaAI(GameState.GameState(gamestate, move, self.playerColor, self.aiColor),alpha,beta,depth-1)
            if cmpVal < bestVal:
                bestVal = cmpVal
                bestMove = move
            if alpha != 'null':
                if cmpVal <= alpha:
                    return bestVal, bestMove
            if beta == 'null' or cmpVal < beta:
                beta = cmpVal
        return bestVal, bestMove

    def heuristic(self,gamestate):
        if self.aiColor == "W":
            return gamestate.getWhiteScore() - gamestate.getBlackScore()
        else:
            return gamestate.getBlackScore() - gamestate.getWhiteScore()

    def mobilityHeuristic(self, gamestate):
        return -1*len(gamestate.validMoves)



    def endGame(self):
        self.gamestates[self.currentState].calculateScore()
        print ''
        print ''
        print '-------------------------'
        print 'Game Over!'
        print '-------------------------'
        print 'Final Score:'
        print 'Black - ' + str(self.gamestates[self.currentState].getBlackScore())
        print 'White - ' + str(self.gamestates[self.currentState].getWhiteScore())
        print ''
        quit()



