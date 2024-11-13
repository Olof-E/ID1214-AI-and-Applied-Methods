# Example file showing a basic pygame "game loop"
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from blocks import blocks

gameState = []
playArea = (10, 20)
screen = None
clock = None
running = False

currBlock = None
rotation = 0
pos = (-1, -1)

def main():
    init()
    gameLoop()
    pygame.quit()


def init():
    global screen, clock, running, dt
    print("Game Started")

    for i in range(playArea[1]):
        gameState.append([])
        for _ in range(playArea[0]):
            gameState[i].append(0)
    
    # print(gameState)

    pygame.init()
    screen = pygame.display.set_mode((1080, 720), pygame.DOUBLEBUF, 32)
    clock = pygame.time.Clock()
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
    global currBlock, rotation, pos
    
    currBlock = blocks[0]
    rotation = 0
    pos = [3, 0]
    
def checkCollision(pos):
    global currBlock, rotation
    rotatedBlock = currBlock[rotation // 90]
    for i in range(4):
        if (pos[0] + rotatedBlock[i][0] < 0 or pos[0] + rotatedBlock[i][0] >= playArea[0]
            or pos[1] + rotatedBlock[i][1] < 0 or pos[1] + rotatedBlock[i][1] >= playArea[1]):
            return True
        
        elif (gameState[pos[1]+rotatedBlock[i][1]][pos[0]+rotatedBlock[i][0]]):
            return True
        
    return False
    
def updateState():
    if(not checkCollision([pos[0], pos[1]+1])):
        pos[1] += 1

def gameLoop():
    global screen, clock, running, dt, currBlock, rotation, pos
    
    passedTime = 0    
    while running:
        
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a and not checkCollision([pos[0] - 1, pos[1]]):
                    pos[0] -= 1
                elif event.key == pygame.K_d and not checkCollision([pos[0] + 1, pos[1]]):
                    pos[0] += 1
                
                if event.key ==  pygame.K_q:
                    rotation = +(rotation + 90) % 360 
                elif event.key ==  pygame.K_e:
                    rotation = +(rotation - 90) % 360 

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")
        
        for i in range(playArea[1]):
            for j in range(playArea[0]):
                draw_rect_alpha(screen, pygame.Color(255, 255*(not gameState[i][j]), 255*(not gameState[i][j]), 255 if gameState[i][j] else 40), pygame.Rect((screen.get_width()-playArea[0]*32)/2.0+32*j, (screen.get_height()-playArea[1]*32)/2.0+32*i,32,32), width=not gameState[i][j])
        
        for i in range(4):
            rotatedBlock = currBlock[rotation // 90]
            topleft = ((screen.get_width()-playArea[0]*32)/2.0, (screen.get_height()-playArea[1]*32)/2.0)
                
            draw_rect_alpha(screen, pygame.Color(255, 0, 0, 255), pygame.Rect(topleft[0]+(pos[0]+rotatedBlock[i][0])*32, topleft[1]+(pos[1]+rotatedBlock[i][1])*32, 32, 32), 0)
        
        pygame.draw.rect(screen, (255,255,255), ((screen.get_width()-playArea[0]*32)/2.0, (screen.get_height()-playArea[1]*32)/2.0, playArea[0]*32, playArea[1]*32), width=2)
        # pygame.draw.circle(screen, "blue", player_pos, 40)

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000
        
        passedTime += dt
        if passedTime > 0.75:
            passedTime = 0
            updateState()


if __name__ == "__main__":
    main()