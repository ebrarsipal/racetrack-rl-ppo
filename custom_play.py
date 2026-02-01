import gymnasium as gym
import highway_env
import pygame
import numpy as np
from stable_baselines3 import PPO
import os

# ==============================
# --------- SETTINGS ----------
# ==============================

WIDTH = 1000
HEIGHT = 700
FPS = 60
CAR_IMAGE_PATH = "car.png"

CAR_EGO_PATH = "images/car1.png"
CAR_OTHER_PATH = "images/car2.png"



# ==============================
# --------- CAMERA ------------
# ==============================

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.offset = pygame.Vector2(0, 0)
        self.zoom = 2.5

    def update(self, target_pos):
        target_pos = pygame.Vector2(float(target_pos[0]), float(target_pos[1]))

        # Zoom hesaba katılarak merkezleme
        desired = target_pos - pygame.Vector2(
            self.width / (2 * self.zoom),
            self.height / (2 * self.zoom)
        )

        self.offset += (desired - self.offset) * 0.08

    def apply(self, pos):
        pos = pygame.Vector2(float(pos[0]), float(pos[1]))
        return (pos - self.offset) * self.zoom

# ==============================
# --------- MAIN --------------
# ==============================

def main():
    

    print("🚀 Custom Renderer Başlatılıyor...")

    # --- ENV ---
    env = gym.make("racetrack-v0")

    env.unwrapped.configure({
        "action": {
            "type": "DiscreteAction",
            "longitudinal": False,
            "lateral": True
        },
        "duration": 120,
        "collision_reward": -100,
        "offroad_terminal": True
    })

    obs, info = env.reset()
    print("✅ Environment hazır.")

    # --- MODEL ---
    model = PPO.load("ppo_racetrack_discrete")
    print("✅ Model yüklendi.")

    # --- PYGAME ---
    pygame.init()
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Custom Racetrack RL Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 22)

    # --- CAMERA ---
    camera = Camera(WIDTH, HEIGHT)
    camera.zoom = 6.0   # 🔥 zoom büyütüldü

    if not os.path.exists(CAR_EGO_PATH):
        print("⚠ Ego car bulunamadı!")
        ego_image = None
    else:
        ego_image = pygame.image.load(CAR_EGO_PATH).convert_alpha()

    if not os.path.exists(CAR_OTHER_PATH):
        print("⚠ Other car bulunamadı!")
        other_image = None
    else:
        other_image = pygame.image.load(CAR_OTHER_PATH).convert_alpha()

    # --- LOAD SPRITE ---
    if not os.path.exists(CAR_IMAGE_PATH):
        print("⚠ car.png bulunamadı! Rectangle çizilecek.")
        car_image = None
    else:
        car_image = pygame.image.load(CAR_IMAGE_PATH).convert_alpha()

    collision_flash = 0
    step_count = 0

    print("🎮 Demo başlıyor...")

    running = True
    while running:

        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # ---- MODEL STEP ----
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        step_count += 1

        ego = env.unwrapped.vehicle
        ego_pos = ego.position

        camera.update(ego_pos)

        # ==============================
        # -------- BACKGROUND ----------
        # ==============================

        for y in range(HEIGHT):
            color = 25 + int(25 * (y / HEIGHT))
            pygame.draw.line(screen, (color, color, color), (0, y), (WIDTH, y))

        # ==============================
        # -------- ROAD DRAW -----------
        # ==============================

        for lane in env.unwrapped.road.network.lanes_list():

            left_points = []
            right_points = []

            lane_width = lane.width

            for s in np.linspace(0, lane.length, 80):

                left = lane.position(s, -lane_width/2)
                right = lane.position(s, lane_width/2)

                left = camera.apply(left)
                right = camera.apply(right)

                left_points.append((left.x, left.y))
                right_points.append((right.x, right.y))

            # Yol polygon
            polygon = left_points + right_points[::-1]

            pygame.draw.polygon(screen, (60, 60, 60), polygon)

            # Kenar çizgileri
            pygame.draw.lines(screen, (255,255,255), False, left_points, 3)
            pygame.draw.lines(screen, (255,255,255), False, right_points, 3)
            center_points = []

            for s in np.linspace(0, lane.length, 30):
                center = lane.position(s, 0)
                center = camera.apply(center)
                center_points.append((center.x, center.y))

            for i in range(0, len(center_points)-1, 4):
                pygame.draw.line(
                    screen,
                    (200,200,200),
                    center_points[i],
                    center_points[i+1],
                    2
                )
        # ==============================
        # -------- VEHICLES ------------
        # ==============================

        for vehicle in env.unwrapped.road.vehicles:

            pos = camera.apply(vehicle.position)

            if vehicle is ego:
                image = ego_image
            else:
                image = other_image

            if image:
                # Ölçek faktörü (oyna burayla)
                scale_factor = 0.08

                rotated = pygame.transform.rotozoom(
                    image,
                    -np.degrees(vehicle.heading),
                    scale_factor
                )

                rect = rotated.get_rect(center=(pos.x, pos.y))
                screen.blit(rotated, rect)

            else:
                pygame.draw.rect(
                    screen,
                    (255,0,0) if vehicle is ego else (0,150,255),
                    (pos.x - 10, pos.y - 5, 20, 10)
                )



        # ==============================
        # -------- COLLISION FLASH -----
        # ==============================

        if ego.crashed:
            collision_flash = 15
            print("💥 Çarpışma!")

        if collision_flash > 0:
            flash_surface = pygame.Surface((WIDTH, HEIGHT))
            flash_surface.set_alpha(90)
            flash_surface.fill((255, 0, 0))
            screen.blit(flash_surface, (0, 0))
            collision_flash -= 1

        # ==============================
        # -------- UI PANEL ------------
        # ==============================

        speed_text = font.render(
            f"Speed: {ego.speed:.2f}",
            True,
            (255, 255, 255)
        )
        reward_text = font.render(
            f"Reward: {reward:.2f}",
            True,
            (255, 255, 255)
        )
        step_text = font.render(
            f"Step: {step_count}",
            True,
            (255, 255, 255)
        )

        screen.blit(speed_text, (20, 20))
        screen.blit(reward_text, (20, 50))
        screen.blit(step_text, (20, 80))

        pygame.display.flip()

        # ==============================
        # -------- LOG -----------------
        # ==============================

        if step_count % 200 == 0:
            print(
                f"Step: {step_count} | "
                f"Speed: {ego.speed:.2f} | "
                f"Reward: {reward:.2f}"
            )

        # ==============================
        # -------- RESET ---------------
        # ==============================

        if done or truncated:
            print("🔁 Episode Reset")
            obs, info = env.reset()

    pygame.quit()
    env.close()
    print("🛑 Program kapandı.")


    

if __name__ == "__main__":
    main()
