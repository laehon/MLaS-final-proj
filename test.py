import serial
import time
import csv
import os

for port in ['COM4', 'COM5']:
    try:
        bt = serial.Serial(port, baudrate=115200, timeout=5)
        print(f"Connected on {port}")
        break
    except:
        print(f"Failed on {port}")

bt.write(b'S')

os.makedirs('./data', exist_ok=True)

def get_next_filename(action):
    i = 1
    while os.path.exists(f"./data/{action}_{i}.csv"):
        i += 1
    return f"./data/{action}_{i}.csv"

def collect_samples(n):
    samples = []
    while len(samples) < n:
        line = bt.readline().decode('utf-8').strip()
        if line:
            samples.append(line.split(','))
    return samples

while True:
    print("\nOptions: test, spike, serve, bump, set, quit")
    choice = input("Action: ").strip().lower()

    if choice == 'quit':
        break

    elif choice == 'test':
        print("Collecting 100 samples...\n")
        for _ in range(100):
            line = bt.readline().decode('utf-8').strip()
            if line:
                print(line)

    elif choice in ('spike', 'serve', 'bump', 'set'):
        print(f"Recording {choice}...")
        samples = collect_samples(500)
        filename = get_next_filename(choice)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
            writer.writerows(samples)
        print(f"Saved {len(samples)} samples to {filename}")

    else:
        print("Unknown action, try again")

bt.close()