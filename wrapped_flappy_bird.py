"""
AUTHOR:         Dhyey Shingala, Aviral Shrivastava, Elham Inamdar
FILENAME:       wrapped_flappy_bird.py
SPECIFICATION:  This file contains the implementation of the environment for the flappy bird game.
FOR:            CS 5392 Reinforcement Learning Section 001
"""
import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS = 100
SCREENWIDTH  = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100 # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()
BACKGROUND_HEIGHT= IMAGES['background'].get_height()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])



class GameState:
    """
    NAME: GameState
    PURPOSE: This class builds the environment for the flappy bird game
    INVARIANTS: The Agent has a memory of previous experiences, a neural network to approximate the Q-function,
    and methods for selecting actions, saving and loading experiences, replacing the target network, and updating the Q-function.
    """
    def __init__(self):
        """
        NAME: __init__
        PURPOSE: Initialize the game state variables
        PRECONDITION: None
        POSTCONDITION: The game state variables are initialized to their default values
        """
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY    =  0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps

    def frame_step(self, input_actions):
        """
        NAME: frame_step
        PARAMETERS: self, input_actions
        PURPOSE: The function performs one frame step in the game
        PRECONDITION: input_actions is a list of two elements, with only one element equal to 1.
        POSTCONDITION: The function returns a tuple containing the current game image, the reward obtained and a boolean
                       value indicating whether the game has ended (terminal).
        """
        #pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                #SOUNDS['wing'].play()

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                #SOUNDS['point'].play()
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash= checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if isCrash:
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            terminal = True
            self.__init__()
            reward = -1

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                   (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        return image_data, reward, terminal

def getRandomPipe():
    """returns a randomly generated pipe"""
    """
    NAME: getRandomPipe
    PURPOSE: Returns a randomly generated pipe consisting of an upper and lower pipe.
    PARAMETERS: None
    PRECONDITION: None
    POSTCONDITION:
    - The function returns a list of two dictionaries, representing the randomly generated pipes.
    - The x and y coordinates of the pipes are set.
    - The gap between the upper and lower pipe is set to a random value from a list of predetermined values.
    """
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs)-1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    """
    NAME: showScore
    PARAMETERS: score, an integer
    PURPOSE: Display the player's current score in the center of the screen during gameplay
    PRECONDITION: This function can be called at any time during gameplay
    POSTCONDITION: The player's current score is displayed on the screen using the 'numbers' image file, with each digit of 
    the score represented by a separate image. The images are displayed in the center of the screen, with a spacing between 
    them based on the width of each image.
    """
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    """
    NAME: checkCrash
    PARAMETERS: player, dictionary with x, y, index, w, h keys
            upperPipes, list of dictionaries with x, y keys
            lowerPipes, list of dictionaries with x, y keys
    PURPOSE: The function checks if the player collides with the base or any of the pipes.
    PRECONDITION: The player, upperPipes, and lowerPipes dictionaries and lists should be properly initialized before 
    calling the function.
    POSTCONDITION: The function returns True if the player collides with the base or any of the pipes, False otherwise.
    """
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    elif player['y'] == 0:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    """
    NAME: pixelCollision
    PARAMETERS: rect1, rect2, hitmask1, hitmask2
    PURPOSE: Checks if two objects have collided with each other
    PRECONDITION: Both objects' rectangular coordinates and hitmasks need to be passed as parameters
    POSTCONDITION: Returns True if the objects collide and False if they do not collide
    """
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False
