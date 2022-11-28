import pygame
import numpy as np
import sys

SCALE = 1 / 3
SCREEN_SIZE = (800, 800)
SCREEN_SIZE = (1920, 1080)
DRAGON_WIDTH, DRAGON_HEIGHT = 326, 326
FLAME_WIDTH, FLAME_HEIGHT = 54, 137

dragon_size = np.array([DRAGON_WIDTH, DRAGON_HEIGHT]) * SCALE
flame_size = np.array((FLAME_WIDTH / 2, FLAME_HEIGHT)) * SCALE
screen = pygame.display.set_mode(SCREEN_SIZE)
screen.fill((255, 255, 255))

timer = pygame.time.Clock()
dragon = pygame.sprite.Sprite()
flame1 = pygame.sprite.Sprite()
flame2 = pygame.sprite.Sprite()
dragon.image = pygame.image.load('dragon_small.png')
flame1.image = pygame.image.load('flame.png')
flame2.image = pygame.image.load('flame.png')

dragon.image = pygame.transform.scale(dragon.image, dragon_size)
flame1.image = pygame.transform.scale(flame1.image, flame_size)
flame2.image = pygame.transform.scale(flame2.image, flame_size)
dragon.orig_image = dragon.image
flame1.orig_image = flame1.image
flame2.orig_image = flame2.image


flame_center1 = np.array([DRAGON_WIDTH / 4, DRAGON_HEIGHT / 2.1]) * SCALE
flame_center2 = np.array([DRAGON_WIDTH * 3 / 4, DRAGON_HEIGHT / 2.1]) * SCALE

dragon_cm = np.array([DRAGON_WIDTH / 2, DRAGON_HEIGHT / 2]) * SCALE


def update_flames():
    flame1_size = flame_size * 1
    flame2_size = flame_size * 0.5
    flame1_corner = (flame_center1[0] - flame1_size[0] / 2, flame_center1[1])
    flame2_corner = (flame_center2[0] - flame2_size[0] / 2, flame_center2[1])
    flame1.image = pygame.transform.scale(flame1.orig_image, flame1_size)
    flame2.image = pygame.transform.scale(flame2.orig_image, flame2_size)
    return flame1_corner, flame2_corner


def display_loop(i):

    screen.fill((255, 255, 255))
    angle = 20
    angle = 1 * i
    # angle = 20
    offset = np.array([100, 200])
    offset = np.array([0, 0])

    # update flames
    flame1_corner, flame2_corner = update_flames()

    # create new dragon image and merge flames onto it
    dragon.image = pygame.transform.rotate(dragon.orig_image, 0)
    dragon.image.blit(
        flame1.image, flame1.image.get_rect(topleft=flame1_corner))
    dragon.image.blit(
        flame2.image, flame2.image.get_rect(topleft=flame2_corner))

    # rotate merged dragon
    dragon.image = pygame.transform.rotate(dragon.image, angle)
    dragon.rect = dragon.image.get_rect()

    # translate merged dragon
    dragon.rect.center = dragon_cm + offset

    # draw everything onto screen
    screen.blit(dragon.image, dragon.rect)

    pygame.image.save(screen, 'test.png')
    pygame.quit()
    sys.exit()


count = -1
while True:
    count += 1
    display_loop(count)
    pygame.display.update()
    timer.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            pygame.quit()
            sys.exit()
