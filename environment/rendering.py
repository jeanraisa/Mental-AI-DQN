import random
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

# Grid and Window settings
GRID_SIZE = 5  # 5x5 Grid
WINDOW_SIZE = 500
CELL_SIZE = WINDOW_SIZE / GRID_SIZE

# Environment Objects
num_obstacles = 5  # Number of obstacles to scatter randomly
obstacles = []

# Generate random obstacles excluding the safe zone and meditation zone
while len(obstacles) < num_obstacles:
    x = random.randint(0, GRID_SIZE - 1)
    y = random.randint(0, GRID_SIZE - 1)
    # Avoid placing obstacles at the safe or meditation zone
    if (x, y) != (GRID_SIZE // 2, GRID_SIZE // 2) and (x, y) != (GRID_SIZE - 1, GRID_SIZE - 1):
        obstacles.append((x, y))

safe_zone = (GRID_SIZE // 2, GRID_SIZE // 2)  # Safe zone in the center
meditation_zone = (GRID_SIZE - 1, GRID_SIZE - 1)  # Meditation zone in the bottom-right corner
patient_position = [0, 0]  # Starting position of the patient

# Movement Path (A longer path to reach the meditation goal)
movement_path = [
    (0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), 
    (4, 2), (4, 1), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1),
    (0, 2), (1, 2), (2, 2), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0),
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4),
    (4, 4)  # This is a long path that will bring the patient to the meditation zone last
]
move_index = 0

# Draw the grid
def draw_grid():
    glColor3f(0, 0, 0)  # Black lines
    glLineWidth(2)
    glBegin(GL_LINES)
    for i in range(GRID_SIZE + 1):
        # Vertical lines
        glVertex2f(i * CELL_SIZE, 0)
        glVertex2f(i * CELL_SIZE, WINDOW_SIZE)

        # Horizontal lines
        glVertex2f(0, i * CELL_SIZE)
        glVertex2f(WINDOW_SIZE, i * CELL_SIZE)
    glEnd()

# Draw squares for obstacles, safe zone, and meditation zone
def draw_square(position, color):
    x, y = position
    glColor3f(*color)
    glBegin(GL_QUADS)
    glVertex2f(x * CELL_SIZE, y * CELL_SIZE)
    glVertex2f((x + 1) * CELL_SIZE, y * CELL_SIZE)
    glVertex2f((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE)
    glVertex2f(x * CELL_SIZE, (y + 1) * CELL_SIZE)
    glEnd()

# Draw the patient as a circle
def draw_circle(position, color):
    x, y = position
    glColor3f(*color)
    glBegin(GL_TRIANGLE_FAN)
    for angle in np.linspace(0, 2 * np.pi, 20):
        glVertex2f((x + 0.5) * CELL_SIZE + np.cos(angle) * CELL_SIZE / 3,
                   (y + 0.5) * CELL_SIZE + np.sin(angle) * CELL_SIZE / 3)
    glEnd()

# Render the environment
def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()

    draw_grid()  # Draw grid lines

    # Draw obstacles (Red)
    for obstacle in obstacles:
        draw_square(obstacle, (1, 0, 0))  # Red

    # Draw safe zone (Yellow)
    draw_square(safe_zone, (1, 1, 0))  # Yellow

    # Draw meditation zone (Green)
    draw_square(meditation_zone, (0, 1, 0))  # Green

    # Draw patient (Blue)
    draw_circle(patient_position, (0, 0, 1))  # Blue

    glutSwapBuffers()

# Move the patient
def move_patient(value):
    global move_index, patient_position

    if move_index < len(movement_path):
        patient_position = movement_path[move_index]
        move_index += 1
        glutPostRedisplay()  # Refresh screen
        glutTimerFunc(500, move_patient, 0)  # Call again in 500ms

# Initialize OpenGL
def init():
    glClearColor(1, 1, 1, 1)  # White background
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, WINDOW_SIZE, 0, WINDOW_SIZE, -1, 1)  # Set 2D mode
    glMatrixMode(GL_MODELVIEW)

# Run the OpenGL visualization
for _ in range(5):
    
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(WINDOW_SIZE, WINDOW_SIZE)
    glutCreateWindow(b"Mental AI - Grid Environment")
    init()
    glutDisplayFunc(display)
    glutTimerFunc(500, move_patient, 0)  # Start movement animation
    glutMainLoop()
