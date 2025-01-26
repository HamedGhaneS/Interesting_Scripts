"""
Ball in Rotating Triangle Animation
Created by: Hamed Ghane
Date: January 26, 2025

This script creates an interactive animation using Pygame:
- A ball bounces inside a rotating triangle using vector-based collision detection
- Controls include Start/Stop/Close buttons and speed adjustment
- Ball movement uses velocity vectors and reflection physics
- Triangle rotates continuously and acts as a boundary
- Speed can be adjusted from 0.5x to 5x using Speed+/- buttons

Key components:
1. Ball physics: Uses Vector2 for position and velocity, with reflection
2. Triangle: Rotates and provides collision boundaries
3. UI: Interactive buttons and speed control
4. Collision detection: Vector-based line-segment collision checking
5. Animation loop: Manages game state and rendering

Required: Python 3.x and Pygame library
"""


import pygame
import math
from pygame import Vector2

pygame.init()
width = 800
height = 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Ball in Triangle")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

# Ball properties
ball_pos = Vector2(width // 2, height // 2)
ball_vel = Vector2(0, 0)  # Start with zero velocity
base_speed = 1
speed_multiplier = 1
ball_radius = 10

# Triangle properties
triangle_size = 100
angle = 0
rotation_speed = 1

# Button properties
button_width = 80
button_height = 30
button_margin = 10
buttons = {
    'start': pygame.Rect(10, 10, button_width, button_height),
    'stop': pygame.Rect(100, 10, button_width, button_height),
    'close': pygame.Rect(190, 10, button_width, button_height),
    'speed+': pygame.Rect(280, 10, button_width, button_height),
    'speed-': pygame.Rect(370, 10, button_width, button_height)
}

# Game state
running = True
animation_running = False

def get_triangle_points():
    points = []
    for i in range(3):
        point_angle = angle + i * 120
        x = width // 2 + triangle_size * math.cos(math.radians(point_angle))
        y = height // 2 + triangle_size * math.sin(math.radians(point_angle))
        points.append(Vector2(x, y))
    return points

def reflect_velocity(normal):
    global ball_vel
    normal = normal.normalize()
    ball_vel = ball_vel - 2 * ball_vel.dot(normal) * normal

def check_line_collision(p1, p2):
    global ball_pos
    ball_to_start = ball_pos - p1
    line_vec = p2 - p1
    line_length = line_vec.length()
    line_unit = line_vec / line_length
    
    projection = ball_to_start.dot(line_unit)
    
    if 0 <= projection <= line_length:
        closest = p1 + line_unit * projection
        dist_vec = closest - ball_pos
        dist = dist_vec.length()
        
        if dist <= ball_radius:
            ball_pos = closest - dist_vec.normalize() * ball_radius
            reflect_velocity(dist_vec)
            return True
    return False

def draw_button(text, rect, active=True):
    color = GREEN if active else GRAY
    pygame.draw.rect(screen, color, rect)
    font = pygame.font.Font(None, 24)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            if buttons['start'].collidepoint(mouse_pos):
                animation_running = True
                if ball_vel.length() == 0:
                    ball_vel = Vector2(base_speed * speed_multiplier, base_speed * speed_multiplier)
            elif buttons['stop'].collidepoint(mouse_pos):
                animation_running = False
            elif buttons['close'].collidepoint(mouse_pos):
                running = False
            elif buttons['speed+'].collidepoint(mouse_pos):
                speed_multiplier = min(speed_multiplier + 0.5, 5.0)
                if animation_running:
                    ball_vel = ball_vel.normalize() * (base_speed * speed_multiplier)
            elif buttons['speed-'].collidepoint(mouse_pos):
                speed_multiplier = max(speed_multiplier - 0.5, 0.5)
                if animation_running:
                    ball_vel = ball_vel.normalize() * (base_speed * speed_multiplier)

    if animation_running:
        ball_pos += ball_vel
        angle += rotation_speed
        points = get_triangle_points()
        for i in range(3):
            if check_line_collision(points[i], points[(i + 1) % 3]):
                break

    # Draw everything
    screen.fill(WHITE)
    
    # Draw buttons
    draw_button('Start', buttons['start'], True)
    draw_button('Stop', buttons['stop'], True)
    draw_button('Close', buttons['close'], True)
    draw_button(f'Speed+', buttons['speed+'], True)
    draw_button(f'Speed-', buttons['speed-'], True)
    
    # Draw speed indicator
    font = pygame.font.Font(None, 24)
    speed_text = font.render(f'Speed: {speed_multiplier:.1f}x', True, BLACK)
    screen.blit(speed_text, (460, 15))
    
    # Draw ball and triangle
    pygame.draw.circle(screen, RED, (int(ball_pos.x), int(ball_pos.y)), ball_radius)
    pygame.draw.polygon(screen, BLACK, [(p.x, p.y) for p in get_triangle_points()], 2)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
