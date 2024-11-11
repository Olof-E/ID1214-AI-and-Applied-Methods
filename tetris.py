# Example file showing a basic pygame "game loop"
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


gameState = []
playArea = (10, 20)
screen = None
clock = None
running = False

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
    
    print(gameState)

    pygame.init()
    screen = pygame.display.set_mode((1080, 1080), pygame.DOUBLEBUF, 32)
    clock = pygame.time.Clock()
    running = True
    dt = 0


def gameLoop():
    global screen, clock, running, dt
    
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("black")
        
        gameState[19][9] = 1
        gameState[18][8] = 1
        gameState[19][8] = 1
        gameState[19][7] = 1
        
        
        for i in range(playArea[1]):
            for j in range(playArea[0]):
                pygame.draw.rect(screen, pygame.Color(255, 255, 255, 2), pygame.Rect((screen.get_width()-playArea[0]*32)/2.0+32*j, (screen.get_height()-playArea[1]*32)/2.0+32*i,32,32), width=not gameState[i][j])
        # pygame.draw.circle(screen, "blue", player_pos, 40)

        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_w]:
        #     player_pos.y -= 300 * dt
        # if keys[pygame.K_s]:
        #     player_pos.y += 300 * dt
        # if keys[pygame.K_a]:
        #     player_pos.x -= 300 * dt
        # if keys[pygame.K_d]:
        #     player_pos.x += 300 * dt

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000


if __name__ == "__main__":
    main()