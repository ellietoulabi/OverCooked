
import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import numpy as np





class ImageAnimator:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.images = self.load_images()

    def load_images(self):
        """Loads and sorts images from the folder path."""
        image_files = sorted([file for file in os.listdir(self.folder_path) if file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))])
        return [Image.open(os.path.join(self.folder_path, img)) for img in image_files]

    def create_gif(self, output_path, duration=0.5):
        """Creates and saves a GIF from the loaded images using imageio."""
        frames = [imageio.imread(os.path.join(self.folder_path, img.filename)) for img in self.images]
        imageio.mimsave(output_path, frames, duration=duration)
        print(f"GIF saved at {output_path}")

    def display_animation(self, interval=500):
        """Displays the animation using matplotlib."""
        fig, ax = plt.subplots()
        im = plt.imshow(self.images[0])

        def update(frame):
            im.set_data(self.images[frame])
            return [im]

        ani = FuncAnimation(fig, update, frames=len(self.images), interval=interval, blit=True)
        plt.show()

    def save_animation_as_gif(self, output_path, interval=500):
        """Creates and saves the animation as a GIF using matplotlib."""
        fig, ax = plt.subplots()
        im = plt.imshow(self.images[0])

        def update(frame):
            im.set_data(self.images[frame])
            return [im]

        ani = FuncAnimation(fig, update, frames=len(self.images), interval=interval, blit=True)
        ani.save(output_path, writer="pillow")
        print(f"GIF saved at {output_path}")


def game_runs_info(data, print_details=False):
    ep_sparse_r_arr = []
    ep_sparse_r_by_agent_arr = []
    ep_shaped_r_arr = []
    ep_shaped_r_by_agent_arr = []
    
    sr =[]
    shr=[]
    
    for game_idx, game in enumerate(data['ep_infos'], start=1):
        print(f"Game {game_idx}")
        print("-" * 20)
        
        # Track end-of-episode rewards
        ep_sparse_r = None
        ep_sparse_r_by_agent = None
        ep_shaped_r = None
        ep_shaped_r_by_agent = None

        for step_idx, step_info in enumerate(game, start=1):
            if print_details:
                print(f"  Timestep {step_idx}")
            
            # Print agent action probabilities
            # for agent_idx, agent_info in enumerate(step_info['agent_infos'], start=1):
            #     action_probs = agent_info['action_probs']
            #     if print_details:
            #         print(f"    Agent {agent_idx} Action Probabilities: {action_probs}")
            
            # Print rewards by agent
            sparse_r = step_info['sparse_r_by_agent']
            shaped_r = step_info['shaped_r_by_agent']
            sr.append(sparse_r)
            shr.append(shaped_r)
            if print_details:
                print(f"    Sparse Reward by Agent: {sparse_r}")
                print(f"    Shaped Reward by Agent: {shaped_r}")
                
            
            # Print additional features if available
            phi_s = step_info.get('phi_s', None)
            phi_s_prime = step_info.get('phi_s_prime', None)
            if print_details:
                if phi_s is not None:
                    print(f"    Phi_s (State Feature): {phi_s}")
                if phi_s_prime is not None:
                    print(f"    Phi_s_prime (Next State Feature): {phi_s_prime}")

            # Check if this step contains end-of-episode information
            if 'episode' in step_info:
                episode_info = step_info['episode']
                ep_sparse_r = episode_info.get('ep_sparse_r', None)
                ep_sparse_r_by_agent = episode_info.get('ep_sparse_r_by_agent', None)
                ep_shaped_r = episode_info.get('ep_shaped_r', None)
                ep_shaped_r_by_agent = episode_info.get('ep_shaped_r_by_agent', None)
                
                ep_sparse_r_arr.append(ep_sparse_r)
                ep_sparse_r_by_agent_arr.append(ep_sparse_r_by_agent)
                ep_shaped_r_arr.append(ep_shaped_r)
                ep_shaped_r_by_agent_arr.append(ep_shaped_r_by_agent)
    
            if print_details:
                print("\n" + "-" * 10 + "\n")
        
        # Print end-of-episode rewards if available
        if ep_sparse_r is not None:
            print(f"  End of Game {game_idx} Summary:")
            print(f"    Total Sparse Reward (ep_sparse_r): {ep_sparse_r}")
            print(f"    Sparse Reward by Agent (ep_sparse_r_by_agent): {ep_sparse_r_by_agent}")
            print(f"    Total Shaped Reward (ep_shaped_r): {ep_shaped_r}")
            print(f"    Shaped Reward by Agent (ep_shaped_r_by_agent): {ep_shaped_r_by_agent}")
        if print_details:
            print("\n" + "=" * 40 + "\n")
    return sr,shr       
    # return ep_sparse_r_arr, ep_sparse_r_by_agent_arr, ep_shaped_r_arr, ep_shaped_r_by_agent_arr

