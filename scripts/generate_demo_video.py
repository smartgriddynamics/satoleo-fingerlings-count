"""
SatoLeo Count - Realistic Fish Simulation & Video Generator
Generates a realistic test video of fish fingerlings swimming in a container/bowl
with natural schooling behavior, tail wiggles, and water reflections.
Enables immediate end-to-end verification of detection, ByteTrack tracking, and counting.

Usage:
    python scripts/generate_demo_video.py --output videos/sample_fish.mp4 --num-fish 12 --duration 10
"""
import argparse
import os
import math
import random
import cv2
import numpy as np

class SimulatedFish:
    def __init__(self, fish_id: int, container_center: tuple, radius: int):
        self.id = fish_id
        self.center = container_center
        self.max_radius = radius - 40
        # Position in polar coordinates
        self.r = random.uniform(20, self.max_radius)
        self.angle = random.uniform(0, 2 * math.pi)
        self.x = self.center[0] + self.r * math.cos(self.angle)
        self.y = self.center[1] + self.r * math.sin(self.angle)
        
        # Velocity and heading
        self.speed = random.uniform(2.5, 4.5)
        self.heading = random.uniform(0, 2 * math.pi)
        self.angular_vel = random.uniform(-0.08, 0.08)
        
        # Appearance
        self.length = random.randint(34, 48)
        self.width = random.randint(12, 18)
        self.tail_phase = random.uniform(0, math.pi * 2)
        self.tail_speed = random.uniform(0.3, 0.6)
        # Fish hue (silvery grey/dark tilapia / carp fingerling tones)
        base_gray = random.randint(35, 70)
        self.color = (base_gray, base_gray + random.randint(5, 15), base_gray + random.randint(10, 25))

    def update(self):
        # Update heading with slight random wandering
        self.angular_vel += random.uniform(-0.02, 0.02)
        self.angular_vel = max(-0.15, min(0.15, self.angular_vel))
        self.heading += self.angular_vel
        
        # Steer away from boundary
        dx = self.x - self.center[0]
        dy = self.y - self.center[1]
        dist_from_center = math.hypot(dx, dy)
        
        if dist_from_center > self.max_radius:
            # Turn back towards center
            target_angle = math.atan2(-dy, -dx)
            angle_diff = (target_angle - self.heading + math.pi) % (2 * math.pi) - math.pi
            self.heading += angle_diff * 0.15
            
        # Move forward
        self.x += self.speed * math.cos(self.heading)
        self.y += self.speed * math.sin(self.heading)
        self.tail_phase += self.tail_speed

    def draw(self, frame: np.ndarray):
        # Calculate fish body points
        h_cos = math.cos(self.heading)
        h_sin = math.sin(self.heading)
        
        # Head, Body center, Tail base
        head_x = int(self.x + (self.length * 0.45) * h_cos)
        head_y = int(self.y + (self.length * 0.45) * h_sin)
        tail_base_x = int(self.x - (self.length * 0.45) * h_cos)
        tail_base_y = int(self.y - (self.length * 0.45) * h_sin)
        
        # Tail fin wag
        wag = math.sin(self.tail_phase) * (self.width * 0.8)
        perp_cos = -h_sin
        perp_sin = h_cos
        tail_tip_x = int(tail_base_x - 12 * h_cos + wag * perp_cos)
        tail_tip_y = int(tail_base_y - 12 * h_sin + wag * perp_sin)

        # Draw fish body (rotated ellipse)
        axes = (int(self.length / 2), int(self.width / 2))
        angle_deg = math.degrees(self.heading)
        cv2.ellipse(frame, (int(self.x), int(self.y)), axes, angle_deg, 0, 360, self.color, -1, cv2.LINE_AA)
        
        # Draw tail fin (triangle)
        fin_pts = np.array([
            (tail_base_x, tail_base_y),
            (tail_tip_x + int(6 * perp_cos), tail_tip_y + int(6 * perp_sin)),
            (tail_tip_x - int(6 * perp_cos), tail_tip_y - int(6 * perp_sin))
        ], np.int32)
        cv2.fillPoly(frame, [fin_pts], (self.color[0] + 15, self.color[1] + 15, self.color[2] + 20), cv2.LINE_AA)

        # Draw small dorsal eye
        eye_x = int(head_x - 4 * h_cos + 3 * perp_cos)
        eye_y = int(head_y - 4 * h_sin + 3 * perp_sin)
        cv2.circle(frame, (eye_x, eye_y), 2, (10, 10, 10), -1, cv2.LINE_AA)

        # Bounding box for verification
        margin = max(self.length, self.width) // 2 + 8
        x1 = max(0, int(self.x - margin))
        y1 = max(0, int(self.y - margin))
        x2 = min(frame.shape[1] - 1, int(self.x + margin))
        y2 = min(frame.shape[0] - 1, int(self.y + margin))
        return [x1, y1, x2, y2]

def generate_simulation_video(output_path: str, num_fish: int = 15, duration_sec: int = 10, fps: int = 30):
    width, height = 800, 600
    total_frames = duration_sec * fps
    center = (width // 2, height // 2)
    bowl_radius = min(width, height) // 2 - 40

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    fish_list = [SimulatedFish(i + 1, center, bowl_radius) for i in range(num_fish)]

    print(f"Generating realistic {num_fish}-fish aquaculture simulation ({duration_sec}s @ {fps}fps)...")

    for f_idx in range(total_frames):
        # Create aquaculture white container / clean water background
        frame = np.full((height, width, 3), 235, dtype=np.uint8)

        # Draw blue water container / bowl
        cv2.circle(frame, center, bowl_radius + 15, (200, 215, 225), -1, cv2.LINE_AA)
        cv2.circle(frame, center, bowl_radius, (240, 235, 215), -1, cv2.LINE_AA)
        # Subtle rim shadow
        cv2.circle(frame, center, bowl_radius, (170, 185, 195), 4, cv2.LINE_AA)

        # Draw subtle dynamic water ripples
        ripple_radius = int(50 + (f_idx % 90) * 2.5)
        alpha_ripple = max(0, 1.0 - (f_idx % 90) / 90.0) * 0.15
        if alpha_ripple > 0:
            overlay = frame.copy()
            cv2.circle(overlay, (center[0] - 40, center[1] - 30), ripple_radius, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha_ripple, frame, 1 - alpha_ripple, 0, frame)

        # Update and draw all fish
        for fish in fish_list:
            fish.update()
            fish.draw(frame)

        out.write(frame)

    out.release()
    print(f"[SUCCESS] Generated simulation video at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fish fingerling simulation video")
    parser.add_argument("--output", type=str, default="videos/sample_fish.mp4", help="Output path")
    parser.add_argument("--num-fish", type=int, default=12, help="Number of fish fingerlings")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    generate_simulation_video(args.output, args.num_fish, args.duration, args.fps)

if __name__ == "__main__":
    main()
