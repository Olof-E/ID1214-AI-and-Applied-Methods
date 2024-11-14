# Example file showing a basic pygame "game loop"
from math import fabs
import os
import random
from turtle import clear

from Vector2 import Vector2
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from blocks import blocks

gameState = []
playArea = Vector2(10, 20)
screen = None
clock = None
running = False

nextPieces = []

currBlock = None
rotation = 0
pos = Vector2(3, -1)

def main():
    init()
    gameLoop()
    pygame.quit()


def init():
    global screen, clock, running, dt, nextPieces
    print("Game Started")

    for i in range(playArea.y):
        gameState.append([])
        for _ in range(playArea.x):
            gameState[i].append(0)
    
    # print(gameState)

    pygame.init()
    screen = pygame.display.set_mode((1080, 720), pygame.DOUBLEBUF, 32)
    clock = pygame.time.Clock()
    
    nextPieces = list(range(len(blocks)))
    random.shuffle(nextPieces)
    
    spawnBlock()
    running = True
    dt = 0


def draw_rect_alpha(surface, color, rect, width):
    shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), width)
    surface.blit(shape_surf, rect)

def move():
    pass

def rotate():
    pass

def spawnBlock():
    global nextPieces, currBlock, rotation, pos
       
    currBlock = blocks[nextPieces.pop(0)]
    
    if len(nextPieces) == 0:
        nextPieces = list(range(len(blocks)))
        random.shuffle(nextPieces)
    
    rotation = 0
    pos = Vector2(3, -1)
    if currBlock == blocks[len(blocks)-1]:
        pos = Vector2(3, 0)
    
def checkCollision(pos):
    global currBlock, rotation
    rotatedBlock = currBlock[rotation // 90]
    for i in range(4):
        if (pos.x + rotatedBlock[i].x < 0 or pos.x + rotatedBlock[i].x >= playArea.x
            or pos.y + rotatedBlock[i].y >= playArea.y):
            return True
        
        elif (gameState[pos.y+rotatedBlock[i].y][pos.x+rotatedBlock[i].x]):
            return True
        
    return 0

def quickDrop():
    currPiece = currBlock
    while(currPiece == currBlock):
        updateState()
    pass

def clearLines():
    global gameState
    
    cleared = []
    for i in range(playArea.y):
        if(sum(gameState[i]) == playArea.x):
            cleared.append(i)
            
    gameState = [row for idx, row in enumerate(gameState) if idx not in cleared]
    for i in range(len(cleared)):
        gameState.insert(0, [0]*playArea.x)
    
def updateState():
    if(not checkCollision(Vector2(pos.x, pos.y+1))):
        pos.y += 1
    else:
        rotatedBlock = currBlock[rotation // 90]
        for i in range(4):
            gameState[pos.y+rotatedBlock[i].y][pos.x+rotatedBlock[i].x] = 1

        clearLines()
        spawnBlock()
            

def gameLoop():
    global screen, clock, running, dt, currBlock, rotation, pos
    
    passedTime = 0    
    while running:
        
        # poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_a and not checkCollision(Vector2(pos.x - 1, pos.y)):
                    pos.x -= 1
                elif event.key == pygame.K_d and not checkCollision(Vector2(pos.x + 1, pos.y)):
                    pos.x += 1
                
                if event.key == pygame.K_SPACE:
                    quickDrop()
                
                if event.key ==  pygame.K_q:
                    rotation = (rotation + 90) % 360 
                elif event.key ==  pygame.K_e:
                    rotation = (rotation - 90) % 360 

        # fill the screen with a color to wipe away anything from last frame
        screen.fill((0, 0, 0))
        
        topleft = Vector2((screen.get_width()-playArea.x*32)/2.0, (screen.get_height()-playArea.y*32)/2.0)
    
        for i in range(playArea.y):
            for j in range(playArea.x):
                draw_rect_alpha(screen, (255, 255*(not gameState[i][j]), 255*(not gameState[i][j]), 255 if gameState[i][j] else 40), pygame.Rect((screen.get_width()-playArea.x*32)/2.0+32*j, (screen.get_height()-playArea.y*32)/2.0+32*i,32,32), width=not gameState[i][j])
        
        for i in range(4):
            rotatedBlock = currBlock[rotation // 90]
                
            draw_rect_alpha(screen, (255, 0, 0, 255), pygame.Rect(topleft.x+(pos.x+rotatedBlock[i].x)*32, topleft.y + (pos.y + rotatedBlock[i].y)*32, 32, 32), 0)
        
        # Draw next piece
        pygame.draw.rect(screen, (255, 255, 255), (topleft.x+playArea.x*32 + 75, topleft.y, 150, 100), width=2)
        for i in range(4):
            if nextPieces:
                rotatedBlock = blocks[nextPieces[0]][0]
                
                draw_rect_alpha(screen, (255, 0, 0, 255), pygame.Rect(topleft.x+playArea.x*32 + 96 + rotatedBlock[i].x*32, topleft.y - 32 + rotatedBlock[i].y*32, 32, 32), 0)
        
        
        # Draw border
        pygame.draw.rect(screen, (255,255,255), (topleft.x, topleft.y, playArea.x*32, playArea.y*32), width=2)

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        dt = clock.tick(60) / 1000
        
        # Drop piece 2 cells / s 
        passedTime += dt
        if passedTime > 0.5:
            passedTime = 0
            updateState()


if __name__ == "__main__":
    main()