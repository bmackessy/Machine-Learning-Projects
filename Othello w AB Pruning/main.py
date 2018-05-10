import Game

def main():
    game = Game.Game()
    game.gamestates[game.currentState].printBoard()
    game.gamestates[game.currentState].printScoreBoard()

    while (True):
        game.makeMove()




if __name__ == '__main__':
    main()