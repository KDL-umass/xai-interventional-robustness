import os
import imageio
import shutil


def make_videos(dir_path, video_filename):
    image_files = os.listdir(dir_path)
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    with imageio.get_writer(video_filename, fps=24, macro_block_size=1) as writer:
        for image in image_files:
            im = imageio.imread(dir_path + image)
            writer.append_data(im)
    writer.close()


def get_image_path(env_name, seed, agent_name):
    return "interventions/results/frames/{}/{}/{}/".format(env_name, seed, agent_name)


def record_image(timestep, env, env_name, seed, agent_name):
    dir_path = get_image_path(env_name, seed, agent_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    env.toybox.save_frame_image(dir_path + "{}.png".format(timestep))