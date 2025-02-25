import random

def uniform_random_sample(num_total_frames, num_clipped_frames):
    clips = []
    interval = num_total_frames / num_clipped_frames
    start = 0
    end = start + interval
    for i in range(num_clipped_frames):
        rand = int(start + random.random() * (end - start))
        if rand < start:
            rand += 1
        if rand > end:
            rand -= 1
        clips.append(rand)
        start = end
        end = end + interval
    return clips



def uniform_sample(num_total_frames, num_clipped_frames):
    clips = []
    interval = num_total_frames / num_clipped_frames # >=1
    start = 0
    end = start + interval
    for i in range(num_clipped_frames):
        clips.append(int(start))
        start = end
        end = end + interval
    return clips
