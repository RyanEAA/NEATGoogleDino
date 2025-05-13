import pygame
import os
import random
import sys
import neat
import math
import matplotlib.pyplot as plt
from neat.reporting import BaseReporter
import matplotlib.animation as animation

# initialize pygame
pygame.init()

# Game Setup Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Importing Imagse
RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]

JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
           pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
           pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]

LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
           pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
           pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

BACKGROUND = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

FONT = pygame.font.Font("freesansbold.ttf", 20)

fitness_history = []
class LivePlotReporter(BaseReporter):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.gen = []
        self.max_fitness = []
        self.avg_fitness = []
        self.min_fitness = []

        self.max_line, = self.ax.plot([], [], 'g-', label='Max Fitness')
        self.avg_line, = self.ax.plot([], [], 'b--', label='Avg Fitness')
        self.min_line, = self.ax.plot([], [], 'r-', label='Min Fitness')

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 10)
        self.ax.set_title("Fitness Over Generations")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.legend()
        plt.ion()
        plt.show()

    def post_evaluate(self, config, population, species, best_genome):
        generation = len(self.gen)

        fitnesses = [genome.fitness for genome in population.values()]
        min_fit = min(fitnesses)
        avg_fit = sum(fitnesses) / len(fitnesses)
        max_fit = max(fitnesses)

        self.gen.append(generation)
        self.min_fitness.append(min_fit)
        self.avg_fitness.append(avg_fit)
        self.max_fitness.append(max_fit)

        self.max_line.set_data(self.gen, self.max_fitness)
        self.avg_line.set_data(self.gen, self.avg_fitness)
        self.min_line.set_data(self.gen, self.min_fitness)

        self.ax.set_xlim(0, max(10, generation + 1))
        self.ax.set_ylim(
            min(self.min_fitness) - 5 if self.min_fitness else -5,
            max(self.max_fitness) + 5 if self.max_fitness else 5
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



# Class for the Dinosaur
class Dinosaur:
    X_POS = 80
    Y_POS = 310
    JUMP_VEL = 8.5

    # Dino Functions
    def __init__(self, img=RUNNING[0]):
        self.image = img # initializing image
        self.dino_run = True
        self.dino_jump = False
        self.jump_vel = self.JUMP_VEL
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height()) # X_POS and Y_POS correspond the top left corner of the image
        self.step_index = 0 # for looping through images
        self.color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))


    def update(self):
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()
        if self.step_index >= 10:
            self.step_index = 0

    def jump(self):
        self.image = JUMPING # updates image to jumping image
        if self.dino_jump: # if dino jumps
            self.rect.y -=  self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel <= -self.JUMP_VEL: 
            self.dino_jump = False
            self.dino_run = True
            self.jump_vel = self.JUMP_VEL


    def run(self):
        self.image = RUNNING[self.step_index // 5] # first 5 shows first image, then second image
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.step_index += 1

    def draw(self, SCREEN): # where the image will be displayed
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))
        pygame.draw.rect(SCREEN, self.color, (self.rect.x, self.rect.y, self.rect.width, self.rect.height), 2)
        for obstacle in obstacles:
            pygame.draw.line(SCREEN, self.color, (self.rect.x + 54, self.rect.y + 12), obstacle.rect.center, 2)

class Obstacle:
    def __init__(self, image, number_of_cacti):
        self.image = image
        self.type = number_of_cacti
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed # moving obstacle to left
        if self.rect.x < -self.rect.width:
            obstacles.pop()
    
    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle):
    def __init__ (self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 325

class LargeCactus(Obstacle):
    def __init__ (self, image, number_of_cacti):
        super().__init__(image, number_of_cacti)
        self.rect.y = 300

def remove(index): # when dino dies we remove it from dino, ge, and nets
    dinosaurs.pop(index)
    ge.pop(index)
    nets.pop(index)

def distance(pos_a, pos_b):
    dx = pos_a[0]-pos_b[0]
    dy = pos_a[1]-pos_b[1]
    return math.sqrt(dx**2+dy**2)

def eval_genomes(genomes, config):
    global game_speed, x_pos_bg, y_pos_bg, obstacles, dinosaurs, ge, nets, points # global variables
    clock = pygame.time.Clock()

    obstacles = [] # stores obtacles that are created
    points = 0
    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 20

    dinosaurs = []
    ge = [] # list of genome
    nets = [] # list for networks

    for genome_id, genome in genomes:
        dinosaurs.append(Dinosaur())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0


    # scoring function
    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = FONT.render(f'Points: {str(points)}', True, (0,0,0))
        SCREEN.blit(text, (950, 50))

    def statistics():
        global dinosaurs, game_speed, ge
        text_1 = FONT.render(f'Dinosaurs Alive: {str(len(dinosaurs))}', True, (0,0,0))
        text_2 = FONT.render(f'Generation: {pop.generation+1}', True, (0,0,0))
        text_3 = FONT.render(f'Game Speed: {str(game_speed)}', True, (0,0,0))

        SCREEN.blit(text_1, (50, 450))
        SCREEN.blit(text_2, (50, 480))
        SCREEN.blit(text_3, (50, 510))

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed

    run = True
    while run:
        for event in pygame.event.get():

            # if user clicks the top right close button
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        SCREEN.fill((255, 255, 255))

        # Reward each dinosaur for surviving each frame
        for i, dinosaur in enumerate(dinosaurs):
            # ge[i].fitness += 0.1  # Small reward for staying alive
            # Optionally, reward based on points (distance traveled)
            ge[i].fitness += points * 0.01  # Scale points to fitness

        for dinosaur in dinosaurs:
            dinosaur.update()
            dinosaur.draw(SCREEN)
        
        if len(dinosaurs) == 0: # if no dinosaurs
            break

        if len(obstacles) == 0:
            rand_int = random.randint(0,1)
            if rand_int ==  0:
                obstacles.append(SmallCactus(SMALL_CACTUS, random.randint(0, 2)))
            elif rand_int == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS, random.randint(0, 2)))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            dead_dinosaurs = []
            for i, dinosaur in enumerate(dinosaurs):
                if dinosaur.rect.colliderect(obstacle.rect):
                    ge[i].fitness -= 0.1  # reward survival
                    dead_dinosaurs.append(i)

            # removing dead dinosaurs
            for i in reversed(dead_dinosaurs):
                remove(i)
        
        # user_input = pygame.key.get_pressed()

        for i, dinosaur in enumerate(dinosaurs):
            output = nets[i].activate((dinosaur.rect.y,
                                       distance((dinosaur.rect.x, dinosaur.rect.y), obstacle.rect.midtop)
                                       ))
            
            if output[0] > 0.5 and dinosaur.rect.y == dinosaur.Y_POS:   
                dinosaur.dino_jump = True
                dinosaur.dino_run = False

        statistics()
        score()
        background()
        clock.tick(30)
        pygame.display.update()


# setup for NEAT
def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    live_plot = LivePlotReporter()
    pop.add_reporter(live_plot)

    # run NEAT for up to 50 generations
    pop.run(eval_genomes, 50)



# main()
if __name__ == '__main__': 
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)

