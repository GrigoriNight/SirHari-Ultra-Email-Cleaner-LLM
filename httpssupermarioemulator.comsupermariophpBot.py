import sched
from time import sleep, time
import win32gui
import win32con
import ctypes
from ctypes import wintypes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from PIL import Image, ImageGrab  # ImageGrab works when minimized!

# === SendInput for background input (WORKS MINIMIZED!) ===
PUL = ctypes.POINTER(ctypes.c_ulong)
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", PUL)]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", wintypes.DWORD),
                ("_input", _INPUT)]

def press_key_background(vk, hold_sec=0.1):
    extra = ctypes.pointer(ctypes.c_ulong(0))
    ii_ = INPUT(type=1, ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=0, time=0, dwExtraInfo=extra))
    ii_up = INPUT(type=1, ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=2, time=0, dwExtraInfo=extra))  # KEYEVENTF_KEYUP
    ctypes.windll.user32.SendInput(1, ctypes.byref(ii_), ctypes.sizeof(ii_))
    sleep(hold_sec)
    ctypes.windll.user32.SendInput(1, ctypes.byref(ii_up), ctypes.sizeof(ii_up))

# === Controls ===
VK_LEFT   = 0x25
VK_RIGHT  = 0x27
VK_X      = 0x58  # Jump
VK_Z      = 0x5A  # Run/Sprint
VK_ENTER  = 0x0D

ACTION_MAP = [
    [],                          # 0: Nothing
    [VK_RIGHT],                  # 1: Right
    [VK_RIGHT, VK_X],            # 2: Right + Jump
    [VK_RIGHT, VK_Z],            # 3: Right + Sprint
    [VK_RIGHT, VK_Z, VK_X],      # 4: Right + Sprint + Jump
]

# === Neural Network (same as before) ===
class MarioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# === Screenshot using ImageGrab (works when minimized!) ===
def grab_game_area():
    # You need to adjust these coordinates once (see instructions below)
    bbox = (100, 100, 740, 580)  # (left, top, right, bottom) - CHANGE THIS!
    img = ImageGrab.grab(bbox)
    img = img.convert('L').resize((84, 84))
    return np.array(img) / 255.0

# === Find Mario window (just to detect it's open) ===
def find_mario_window():
    def enum(h, _):
        if win32gui.IsWindowVisible(h):
            t = win32gui.GetWindowText(h).lower()
            if "mario" in t or "supermario" in t:
                print(f"Mario window found: {win32gui.GetWindowText(h)}")
                return h
    return win32gui.EnumWindows(enum, None)

# === Main AI Bot (runs in background!) ===
def main():
    print("Super Mario AI Bot (Background + Minimized Mode)")
    print("Make sure the game is open and visible once to set coordinates!")

    # Wait a moment for you to focus the game
    print("You have 10 seconds to click on the game and make it visible...")
    sleep(10)

    # Capture once to set bbox (you'll see the coordinates printed)
    print("Capturing screen area... (adjust bbox in code if needed)")
    img = ImageGrab.grab()
    print(f"Full screen size: {img.size}")
    print("Edit the 'bbox' variable in the script to match your game area!")
    print("Example: bbox = (100, 100, 740, 580)")

    policy_net = MarioNet()
    target_net = MarioNet()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)
    memory = deque(maxlen=10000)
    steps = 0
    eps = 1.0

    stack = deque(maxlen=4)
    for _ in range(4):
        stack.append(torch.zeros(84, 84))

    print("Bot starting... You can now minimize Chrome! AI is running in background.")

    # Start game
    press_key_background(VK_ENTER, 0.1)
    sleep(2)

    while True:
        frame = torch.tensor(grab_game_area(), dtype=torch.float32).unsqueeze(0)
        stack.append(frame)
        state = torch.cat(list(stack)).unsqueeze(0)

        if random.random() > eps:
            with torch.no_grad():
                action = policy_net(state).argmax().item()
        else:
            action = random.randrange(5)

        eps = max(0.05, eps * 0.995)
        steps += 1

        # Execute action in background
        for vk in ACTION_MAP[action]:
            press_key_background(vk, 0.12)

        # Small reward for surviving
        reward = 0.1

        # Next state
        next_frame = torch.tensor(grab_game_area(), dtype=torch.float32).unsqueeze(0)
        stack.append(next_frame)
        next_state = torch.cat(list(stack)).unsqueeze(0)

        memory.append((state, torch.tensor([[action]]), next_state, torch.tensor([reward])))

        # Train
        if len(memory) >= 32:
            batch = random.sample(memory, 32)
            states, actions, next_states, rewards = zip(*batch)
            states = torch.cat(states)
            actions = torch.cat(actions)
            next_states = torch.cat(next_states)
            rewards = torch.cat(rewards)

            q = policy_net(states).gather(1, actions)
            next_q = target_net(next_states).max(1)[0].detach()
            target = rewards + 0.99 * next_q

            loss = F.smooth_l1_loss(q, target.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if steps % 1000 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Step {steps} | Epsilon: {eps:.3f} | Memory: {len(memory)}")

        sleep(0.05)  # ~20 FPS

if __name__ == "__main__":
    main()