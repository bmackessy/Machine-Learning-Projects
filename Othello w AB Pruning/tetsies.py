def nwCorner(self, boardArr, i, j):
    a = i

    while a >= 0:
        if boardArr[a][j] == ' ':
        #if gamestate.boardArr[a][j] != ' ':
            return False
        a -= 1

    b = j
    while b >= 0:
        if boardArr[i][b] == ' ':
        #if gamestate.boardArr[i][b] != ' ':
            return False
        b -= 1

    a = i
    b = j
    while a >= 0 and b >= 0:
        if boardArr[a][b] == ' ':
        #if gamestate.boardArr[a][b] != ' ':
            return False
        b -= 1
        a -= 1

    return True

def neCorner(self, boardArr, i, j):
    a = i

    while a <= 7:
        if boardArr[a][j] == ' ':
        #if gamestate.boardArr[a][j] != ' ':
            return False
        a += 1

    b = j
    while b >= 0:
        if boardArr[i][b] == ' ':
        #if gamestate.boardArr[i][b] != ' ':
            return False
        b -= 1

    a = i
    b = j
    while a <= 7 and b >= 0:
        if boardArr[a][b] == ' ':
        #if gamestate.boardArr[a][b] != ' ':
            return False
        b -= 1
        a += 1

    return True


def seCorner(self, boardArr, i, j):
    a = i

    while a <= 7:
        if boardArr[a][j] == ' ':
        #if gamestate.boardArr[a][j] != ' ':
            return False
        a += 1

    b = j
    while b <= 7:
        if boardArr[i][b] == ' ':
        #if gamestate.boardArr[i][b] != ' ':
            return False
        b += 1

    a = i
    b = j
    while a <= 7 and b <= 0:
        if boardArr[a][b] == ' ':
        #if gamestate.boardArr[a][b] != ' ':
            return False
        b += 1
        a += 1

    return True


def swCorner(self, boardArr, i, j):
    a = i

    while a >= 0:
        if boardArr[a][j] == ' ':
        #if gamestate.boardArr[a][j] != ' ':
            return False
        a -= 1

    b = j
    while b <= 7:
        if boardArr[i][b] == ' ':
        #if gamestate.boardArr[i][b] != ' ':
            return False
        b += 1

    a = i
    b = j
    while a >= 0 and b <= 7:
        if boardArr[a][b] == ' ':
        #if gamestate.boardArr[a][b] != ' ':
            return False
        b += 1
        a -= 1

    return True


def isStable(self, boardArr, i, j):
    if nwCorner(boardArr, i, j):
        return True

    elif neCorner(boardArr, i, j):
        return True

    elif seCorner(boardArr, i, j):
        return True

    elif swCorner(boardArr, i, j):
        return True

    else:
        return False




boardArr = [[' ','B',' ','B','B','B','B','B'],
            ['B','B',' ','B','B','B','B','B'],
            ['B', ' ', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', ' ', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', ' ', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', ' ', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
            ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']]



print isStable(boardArr, 1, 1)

