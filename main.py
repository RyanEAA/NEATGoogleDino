import pygame
import os
import random
import sys
import neat
import math
import matplotlib.pyplot as plt
from neat.reporting import BaseReporter
import visualize
import io
from PIL import Image

# Initialize pygame
pygame.init()

# Game Setup Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Importing Images
RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]

JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))

CRAWLING = [pygame.image.load(os.path.join("Assets/Dino", "DinoCrawl1.png")),
            pygame.image.load(os.path.join("Assets/Dino", "DinoCrawl2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]

LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

PTERODACTYL = [pygame.image.load(os.path.join("Assets/Pterodactyl", "Pterodactyl1.png")),
               pygame.image.load(os.path.join("Assets/Pterodactyl", "Pterodactyl2.png"))]

BACKGROUND = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))
FONT = pygame.font.Font("freesansbold.ttf", 20)

class LivePlotReporter(BaseReporter):
    def __init__(self, config):
        self.fig, self.ax = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        self.fig2.canvas.manager.set_window_title("Best Genome")
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
        dot = visualize.draw_net(config, best_genome, view=False, show_disabled=True, prune_unused=False)
        png_data = dot.pipe(format='png')
        image = Image.open(io.BytesIO(png_data))
        self.ax2.imshow(image)
        self.ax2.axis('off')

class Dinosaur:
    X_POS = 80
    Y_POS = 310
    JUMP_VEL = 8.5

    def __init__(self, img=RUNNING[0]):
        self.image = img
        self.dino_run = True
        self.dino_jump = False
        self.dino_crawl = False
        self.jump_vel = self.JUMP_VEL
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, img.get_width(), img.get_height())
        self.step_index = 0
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.crawling_rect_size = CRAWLING[0].get_rect().size

    def update(self):
        if self.dino_jump:
            self.jump()
        elif self.dino_crawl and self.rect.y == self.Y_POS:
            self.crawl()
        else:
            self.run()
        if self.step_index >= 10:
            self.step_index = 0

    def jump(self):
        self.image = JUMPING
        if self.dino_jump:
            self.rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel <= -self.JUMP_VEL:
            self.dino_jump = False
            self.dino_run = True
            self.dino_crawl = False
            self.jump_vel = self.JUMP_VEL
            self.rect.size = RUNNING[0].get_rect().size

    def crawl(self):
        self.image = CRAWLING[self.step_index // 5]
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS + 30
        self.rect.size = self.crawling_rect_size
        self.step_index += 1

    def run(self):
        self.image = RUNNING[self.step_index // 5]
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.rect.size = RUNNING[0].get_rect().size
        self.step_index += 1

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))
        pygame.draw.rect(SCREEN, self.color, (self.rect.x, self.rect.y, self.rect.width, self.rect.height), 2)
        for obstacle in obstacles:
            pygame.draw.line(SCREEN, self.color, (self.rect.x + 54, self.rect.y + 12), obstacle.rect.center, 2)

class Obstacle:
    def __init__(self, image):
        self.image = image
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image)
        self.rect.y = 325

class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image)
        self.rect.y = 300

class Pterodactyl(Obstacle):
    HEIGHTS = [210, 240, 325]

    def __init__(self, images):
        self.images = images
        self.frame_index = 0
        self.type = random.randint(0, 2)
        self.rect = self.images[0].get_rect()
        self.rect.x = SCREEN_WIDTH
        self.rect.y = self.HEIGHTS[self.type]
        self.animation_counter = 0

    def draw(self, SCREEN):
        self.animation_counter += 1
        if self.animation_counter % 5 == 0:
            self.frame_index = (self.frame_index + 1) % len(self.images)
        SCREEN.blit(self.images[self.frame_index], self.rect)

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

def remove(index):
    dinosaurs.pop(index)
    ge.pop(index)
    nets.pop(index)

def distance(pos_a, pos_b):
    dx = pos_a[0] - pos_b[0]
    dy = pos_a[1] - pos_b[1]
    return math.sqrt(dx ** 2 + dy ** 2)

def eval_genomes(genomes, config):
    global game_speed, x_pos_bg, y_pos_bg, obstacles, dinosaurs, ge, nets, points
    clock = pygame.time.Clock()

    obstacles = []
    points = 0
    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 20

    dinosaurs = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        dinosaurs.append(Dinosaur())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = FONT.render(f'Points: {str(points)}', True, (0, 0, 0))
        SCREEN.blit(text, (950, 50))

    def statistics():
        global dinosaurs, game_speed, ge
        text_1 = FONT.render(f'Dinosaurs Alive: {str(len(dinosaurs))}', True, (0, 0, 0))
        text_2 = FONT.render(f'Generation: {pop.generation + 1}', True, (0, 0, 0))
        text_3 = FONT.render(f'Game Speed: {str(game_speed)}', True, (0, 0, 0))
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
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.fill((255, 255, 255))

        for i, dinosaur in enumerate(dinosaurs):
            ge[i].fitness += points * 0.01

        for dinosaur in dinosaurs:
            dinosaur.update()
            dinosaur.draw(SCREEN)

        if len(dinosaurs) == 0:
            break

        if len(obstacles) == 0:
            rand_int = random.randint(0, 2)
            if rand_int == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif rand_int == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            else:
                obstacles.append(Pterodactyl(PTERODACTYL))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            dead_dinosaurs = []
            for i, dinosaur in enumerate(dinosaurs):
                if dinosaur.rect.colliderect(obstacle.rect):
                    ge[i].fitness -= 0.1
                    dead_dinosaurs.append(i)

            for i in reversed(dead_dinosaurs):
                remove(i)

        for i, dinosaur in enumerate(dinosaurs):
            if obstacles:
                obstacle = obstacles[0]
                inputs = (
                    dinosaur.rect.y,
                    distance((dinosaur.rect.x, dinosaur.rect.y), obstacle.rect.midtop),
                    obstacle.rect.width,
                    obstacle.rect.height,
                    game_speed,
                    obstacle.rect.y,
                    obstacle.rect.bottom  # Added obstacle bottom
                )
            else:
                inputs = (
                    dinosaur.rect.y,
                    1000.0,  # Large distance
                    0.0,     # Default width
                    0.0,     # Default height
                    game_speed,
                    0.0,     # Default obstacle y
                    0.0      # Default obstacle bottom
                )
            print(obstacle.rect.y, obstacle.rect.bottom)
            outputs = nets[i].activate(inputs)
            if outputs[0] > 0.5 and dinosaur.rect.y == dinosaur.Y_POS:
                dinosaur.dino_jump = True
                dinosaur.dino_run = False
                dinosaur.dino_crawl = False
            elif outputs[1] > 0.5 and dinosaur.rect.y == dinosaur.Y_POS:
                dinosaur.dino_crawl = True
                dinosaur.dino_run = False
                dinosaur.dino_jump = False
            elif dinosaur.rect.y == dinosaur.Y_POS:
                dinosaur.dino_run = True
                dinosaur.dino_crawl = False
                dinosaur.dino_jump = False
            else:
                pass

        statistics()
        score()
        background()
        clock.tick(30)
        pygame.display.update()

def run(config_path):
    global pop
    try:
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        sys.exit(1)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    live_plot = LivePlotReporter(config)
    pop.add_reporter(live_plot)
    pop.run(eval_genomes, 100)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)