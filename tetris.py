# Example file showing a basic pygame "game loop"
import linecache
import os
from turtle import clear
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from blocks import blocks
from Vector2 import Vector2

import random

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 250, 0)
PURPLE = (255, 0, 255)
ORANGE = (220, 100, 20)
CYAN = (0, 255, 255)

colors = [CYAN, BLUE, ORANGE, YELLOW, GREEN, PURPLE, RED]


    # gameState = []
    # playArea = Vector2(10, 20)
    # screen = None
    # clock = None
    # running = False

    # nextPieces = []

    # currBlock = -1
    # heldPiece = -1
    # heldAvailable = True

    # rotation = 0
    # pos = Vector2(3, -1)

    # score = 0
    # level = 0
    # lineClears = 0

    # paused = False
    # quickDropping = False

class Tetris():

    def __init__(self):
        self.gameState = []
        self.playArea = Vector2(10, 20)
        self.screen = None
        self.clock = None
        self.running = False

        self.nextPieces = []

        self.currBlock = -1
        self.heldPiece = -1
        self.heldAvailable = True

        self.rotation = 0
        self.pos = Vector2(3, -1)

        self.score = 0
        self.level = 0
        self.lineClears = 0

        self.paused = False
        self.quickDropping = False
        
        self.dead = False
        
        
    def start(self, mode):
        pygame.init()
        self.init()
        if(mode):
            self.gameLoop()
        
    def quitGame(self):
        pygame.quit()


    def init(self):
        print("Game Started")
        self.dead = False
        
        self.gameState.clear()
        for i in range(self.playArea.y):
            self.gameState.append([])
            for _ in range(self.playArea.x):
                self.gameState[i].append(Vector2(0, -1))
        
        self.screen = pygame.display.set_mode((1080, 720), pygame.DOUBLEBUF, 32)
        self.clock = pygame.time.Clock()
        
        self.nextPieces = list(range(len(blocks)))
        self.heldPiece = -1
        self.heldAvailable = True
            
        random.shuffle(self.nextPieces)
        
        self.spawnBlock()
            
        self.running = True
        self.dt = 0
        self.passedTime = 0
        
        #  [ [Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec] 
        #    [Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec, Vec]
        # ]
        
        
    def newGame(self):
        print("New game started")
        
        self.dead = False
        
        self.gameState.clear()
        for i in range(self.playArea.y):
            self.gameState.append([])
            for _ in range(self.playArea.x):
                self.gameState[i].append(Vector2(0, -1))
        
        
        self.rotation = 0
        self.pos = Vector2(3, -1)

        self.score = 0
        self.level = 0
        self.lineClears = 0         

        self.nextPieces = list(range(len(blocks)))
        self.heldPiece = -1
        self.heldAvailable = True
            
        random.shuffle(self.nextPieces)
        
        self.spawnBlock()


        self.dt = 0
        self.passedTime = 0



    def draw_rect_alpha(self, surface, color, rect, width):
        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), width)
        surface.blit(shape_surf, rect)

    def move(self, direction):
        if not self.checkCollision(Vector2(self.pos.x + direction, self.pos.y), self.rotation):
            self.pos.x += direction

    def rotate(self, direction):
        if not self.checkCollision(Vector2(self.pos.x, self.pos.y), (self.rotation + 90 * direction) % 360):
            self.rotation = (self.rotation + 90 * direction) % 360

    def spawnBlock(self): 
        self.currBlock = self.nextPieces.pop(0)
        
        if len(self.nextPieces) == 0:
            self.nextPieces = list(range(len(blocks)))
            random.shuffle(self.nextPieces)
        
        self.rotation = 0
        self.pos = Vector2(3, -1)
        if blocks[self.currBlock] == blocks[len(blocks)-1]:
            self.pos = Vector2(3, 0)
            
        if(self.checkCollision(self.pos, self.rotation)):
            self.dead = True
            print("You Lost")

    def quickDrop(self):        
        while(self.updateState()):
            pass
        
    def switchHeld(self):
        self.heldAvailable = False
        if self.heldPiece == -1:
            self.heldPiece = self.currBlock
            self.spawnBlock()
        else:
            temp = self.heldPiece
            self.heldPiece = self.currBlock
            self.currBlock = temp
            self.rotation = 0
            self.pos = Vector2(3, -1)
            if blocks[self.currBlock] == blocks[len(blocks)-1]:
                self.pos = Vector2(3, 0)
        
        
    def checkCollision(self, pos, rot):
        rotatedBlock = blocks[self.currBlock][rot // 90]
        for i in range(4):
            if (pos.x + rotatedBlock[i].x < 0 or pos.x + rotatedBlock[i].x >= self.playArea.x
                or pos.y + rotatedBlock[i].y >= self.playArea.y):
                return True
            
            elif (self.gameState[pos.y+rotatedBlock[i].y][pos.x+rotatedBlock[i].x].x):
                return True
            
        return 0
        
    def clearLines(self):        
        cleared = []
        for i in range(self.playArea.y):
            if(sum(cell.x for cell in self.gameState[i]) == self.playArea.x):
                cleared.append(i)

        self.gameState = [row for idx, row in enumerate(self.gameState) if idx not in cleared]
        for i in range(len(cleared)):
            self.gameState.insert(0, [Vector2(0, -1)]*self.playArea.x)    
        
        self.lineClears += len(cleared)
        if(self.lineClears > 10):
            self.lineClears = 0
            self.level += 1
        
        if(len(cleared) == 1):
            self.score += 40 * (self.level + 1)
        elif(len(cleared) == 2):
            self.score += 100 * (self.level + 1)
        elif(len(cleared) == 3):
            self.score += 300 * (self.level + 1)
        elif(len(cleared) == 4):
            self.score += 1200 * (self.level + 1)
            
        print(f"Score: {self.score}")
        
    def updateState(self):        
        if(not self.checkCollision(Vector2(self.pos.x, self.pos.y+1), self.rotation)):
            self.pos.y += 1
            return True
        else:
            rotatedBlock = blocks[self.currBlock][self.rotation // 90]
            for i in range(4):
                self.gameState[self.pos.y+rotatedBlock[i].y][self.pos.x+rotatedBlock[i].x] = Vector2(1, self.currBlock)
            
            self.clearLines()
            self.spawnBlock()
            self.heldAvailable = True
            return False
    
    def step(self):
         # poll for events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and self.dead:
                        self.newGame()
                    
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                    
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_a and not self.checkCollision(Vector2(self.pos.x - 1, self.pos.y), self.rotation):
                        self.pos.x -= 1
                    elif event.key == pygame.K_d and not self.checkCollision(Vector2(self.pos.x + 1, self.pos.y), self.rotation):
                        self.pos.x += 1
                    elif event.key == pygame.K_s:
                        self.quickDrop()
                    
                    if event.key == pygame.K_SPACE and self.heldAvailable:
                        self.switchHeld()
                                            
                    if event.key ==  pygame.K_q and not self.checkCollision(Vector2(self.pos.x, self.pos.y), (self.rotation + 90) % 360):
                        self.rotation = (self.rotation + 90) % 360 
                    elif event.key ==  pygame.K_e and not self.checkCollision(Vector2(self.pos.x, self.pos.y), (self.rotation - 90) % 360):
                        self.rotation = (self.rotation - 90) % 360 

            # fill the screen with a color to wipe away anything from last frame
            self.screen.fill((0, 0, 0))
            
            topleft = Vector2((self.screen.get_width()-self.playArea.x*32)/2.0, (self.screen.get_height()-self.playArea.y*32)/2.0)
        
            for i in range(self.playArea.y):
                for j in range(self.playArea.x):
                    self.draw_rect_alpha(self.screen, colors[self.gameState[i][j].y] if self.gameState[i][j].x else (255, 255, 255, 40), pygame.Rect(topleft.x+32*j, topleft.y+32*i,32,32), not(self.gameState[i][j].x))
            
            for i in range(4):
                rotatedBlock = blocks[self.currBlock][self.rotation // 90]
                    
                pygame.draw.rect(self.screen, colors[self.currBlock], pygame.Rect(topleft.x+(self.pos.x+rotatedBlock[i].x)*32, topleft.y + (self.pos.y + rotatedBlock[i].y)*32, 32, 32))
            
            # Draw next piece
            pygame.draw.rect(self.screen, WHITE, (topleft.x+self.playArea.x*32 + 75, topleft.y, 6*32, 4*32), width=2)
            for i in range(4):
                    rotatedBlock = blocks[self.nextPieces[0]][0]
                    self.draw_rect_alpha(self.screen, colors[self.nextPieces[0]], pygame.Rect(topleft.x+self.playArea.x*32 + 75 + 48 + rotatedBlock[i].x*32, topleft.y - 32 + rotatedBlock[i].y*32, 32, 32), 0)
                    
            
            # Draw held piece
            pygame.draw.rect(self.screen, WHITE, (topleft.x- 6*32 - 75, topleft.y, 6*32, 4*32), width=2)
            if self.heldPiece != -1:
                for i in range(4):
                    rotatedBlock = blocks[self.heldPiece][0]
                    self.draw_rect_alpha(self.screen, colors[self.heldPiece], pygame.Rect(topleft.x - 6*32 - 75 + 48 + rotatedBlock[i].x*32, topleft.y - 32 + rotatedBlock[i].y*32, 32, 32), 0)
            
            
            # Draw border
            pygame.draw.rect(self.screen, WHITE, (topleft.x, topleft.y, self.playArea.x*32, self.playArea.y*32), width=2)

            # flip() the display to put your work on screen
            pygame.display.flip()

            # limits FPS to 60
            self.dt = self.clock.tick(60) / 1000
            
            # Drop piece 2 cells / s 
            self.passedTime += self.dt
            if self.passedTime > 0.5 and not self.paused and not self.dead:
                self.passedTime = 0
                self.updateState()

    def gameLoop(self):     
        while self.running:
            self.step()
            
        self.quitGame()
        


if __name__ == "__main__":
    tetris = Tetris()
    tetris.start(True)