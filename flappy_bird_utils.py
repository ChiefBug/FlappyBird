"""
AUTHOR:         Dhyey Shingala, Aviral Shrivastava, Elham Inamdar
FILENAME:       flappy_bird_utils.py
SPECIFICATION:  This file has the utils to run the environment and the hitboxes for the game.
FOR:            CS 5392 Reinforcement Learning Section 001
"""
import pygame
import sys
def load():
    """
    NAME: load
    PARAMETERS: None
    PURPOSE: Set path of the sprites for the game environments.
    PRECONDITION: You have the sprites stored in a local folder.
    POSTCONDITION: The sprite paths are set and loaded.
    """
    # path of player with different states
    PLAYER_PATH = (
            'C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/redbird-midflap.png',
            'C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/redbird-midflap.png',
            'C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/redbird-midflap.png'
    )

    # path of background
    BACKGROUND_PATH = 'C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/background-day.png'

    # path of pipe
    PIPE_PATH = 'C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/pipe-green.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/0.png').convert_alpha(),
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/1.png').convert_alpha(),
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/2.png').convert_alpha(),
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/3.png').convert_alpha(),
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/4.png').convert_alpha(),
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/5.png').convert_alpha(),
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/6.png').convert_alpha(),
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/7.png').convert_alpha(),
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/8.png').convert_alpha(),
        pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('C:/Users/shing/Documents/Semester_4/Reinforcement Learning/FlappyDQN-main/assets/audio/wing' + soundExt)

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    """
    NAME: getHitmask
    PARAMETERS: image, a pygame.Surface object representing an image
    PURPOSE: This function returns a hitmask using an image's alpha
    PRECONDITION: The image parameter must be a valid pygame.Surface object
    POSTCONDITION: Returns a 2D list of boolean values representing the hitmask
    """
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
